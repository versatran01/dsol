#include "sv/dsol/adjust.h"

#include <mutex>

#include "sv/util/logging.h"
#include "sv/util/tbb.h"

namespace sv::dsol {

using Vector2i = Eigen::Vector2i;
using Vector3i = Eigen::Vector3i;
using Vector3d = Eigen::Vector3d;

/// ============================================================================
AdjustCost::AdjustCost(int level,
                       const Camera& camera_scaled,
                       const Keyframe& kf0,
                       const Keyframe& kf1,
                       const Vector2i& iandj,
                       FramePointHessian& block,
                       const DirectCostCfg& cost_cfg,
                       double delta_info)
    : DirectCost(level, camera_scaled, kf1, cost_cfg),
      ij{iandj},
      pblock{CHECK_NOTNULL(&block)},
      dinfo{delta_info} {
  const auto ep0 = kf0.GetEvaluationPoint();
  const auto ep1 = kf1.GetEvaluationPoint();
  ExtractState(ep0, ep1, T10, eas, bs);
  const auto fe0 = kf0.GetFirstEstimate();
  const auto fe1 = kf1.GetFirstEstimate();
  ExtractState(fe0, fe1, T10_fej, eas_fej, bs_fej);
}

FrameHessian2 AdjustCost::Build(const FramePointGrid& points0,
                                const PatchGrid& patches0,
                                int gsize) const noexcept {
  return ParallelReduce(
      {0, points0.rows(), gsize},
      FrameHessian2{ij},
      [&](int gr, FrameHessian2& hess_ij) {
        for (int gc = 0; gc < points0.cols(); ++gc) {
          const auto& point0 = points0.at(gr, gc);
          // Here we rely on hid instead of point status. This is because we
          // want to give bad points a chance to become good again during the
          // optimization, rather than removing them too early.
          // If a point stays bad (info < 0) after this round of adjustment,
          // then it will be permanently disabled.
          if (point0.HidBad()) continue;

          // Remeber that even with a good point we might have a bad patch
          // in a low res image due to down scaling and too close to border. So
          // we need to check if this patch is ok to use.
          const auto& patch0 = patches0.at(gr, gc);
          if (patch0.Bad()) continue;

          // res is -1 for outlier, 0 for oob and 1 for inlier
          const auto res = WarpPatch(patch0, point0, hess_ij);

          point0.UpdateInfo(res * dinfo);
        }  // gc
      },   // gr
      std::plus<>{});
}

int AdjustCost::WarpPatch(const Patch& patch0,
                          const FramePoint& point0,
                          FrameHessian2& hess01) const noexcept {
  // First construct all uvs of this patch
  const auto uv0 = ScaleUv(point0.uv(), camera.scale());
  const auto uv0s = (Patch::offsets().colwise() + uv0).eval();

  // Then do a batch warp for all pixels in patch
  // see AlignCost::WarpPatch for more details
  const auto q0 = point0.idepth();
  const auto nh0s = camera.Backward(uv0s);
  const auto pt1s = TransformScaled(T10, nh0s, q0);

  Patch::Point2dArray px1s{};
  Eigen::Map<Matrix2Kd> uv1s(&px1s[0].x);
  uv1s = camera.Forward(pt1s);

  // p1 = p1' / q0
  // z1 = z1' / q0
  // q1 = 1 / z1 = q0 / z1'
  const Vector3d pt1 = pt1s.col(0);
  const auto z1 = pt1.z() / q0;

  // Criteria for updating point info
  // If its projection is bad (OOB or depth too small), we maintain its info
  // If it is being used in UpdateHessian, we increase its info
  // If it is considered an outlier during UpdateHessian, we decrease its info
  int res = 0;
  if (!(z1 > cfg.min_depth)) return res;

  Patch patch1;

  // If no OOB, we update hessian for this patch
  if (!Patch::IsAnyOut(gray1l, px1s)) {
    patch1.ExtractFast(gray1l, px1s);

    // Left image cam_ind = 0
    const auto good = UpdateHess(patch0, point0, patch1, pt1, hess01, 0);
    res = good ? 1 : -1;
  }

  if (cfg.stereo) {
    ApplyDisp(uv1s, pt1s.row(2).array(), q0);

    // Check for OOB
    if (!Patch::IsAnyOut(gray1r, px1s)) {
      // Extract pixel intensity from right image, can reuse storage
      patch1.ExtractFast(gray1r, px1s);
      // Right image cam_ind = 1
      UpdateHess(patch0, point0, patch1, pt1, hess01, 1);
    }
  }

  return res;
}

auto AdjustCost::CalcJacGeo(const FramePoint& point0,
                            const Vector3d& pt1l,
                            int cam_ind) const noexcept -> JacGeo {
  const auto q0 = point0.idepth();
  const auto b = cam_ind * camera.baseline();

  // b = 0 means projecting to left image
  // du1_dp1 needs pt1 in right frame if cam_ind is 1, because it is the
  // evaluation point
  Vector3d pt1 = pt1l;
  pt1.x() -= b * q0;

  // u = K * n, ux = fx * x + cx, uy = fy * y + cy
  // p = [x, y, 1]^T
  // n = proj(p) = [x/z, y/z]^T

  // du_dp = du_dn * dn_dp
  // du_dn = [fx, 0]
  //         [0, fy]
  // dn_dp = [1/z,  0, -x/z^2]
  //         [0,  1/z, -y/z^2]
  MatrixMNd<2, 3> du1_dp1 = DprojDpoint(pt1);
  du1_dp1.array().colwise() *= camera.fxy();

  JacGeo j_geo;
  MatrixMNd<3, 6> dp_dx;  // storage for both frame 0 and 1

  // These need to be evaluated at x = 0 (FEJ)
  // p1  = R10 * n0 / q0 + t_1_0    - b
  // p1' = R10 * n0      + t10 * q0 - b * q0

  // dp1_dr0 = -R10 * [n]x
  dp_dx.leftCols<3>().noalias() = -T10_fej.linear() * Hat3d(point0.nh());
  // dp1_dt0 = -q0 * R10
  dp_dx.rightCols<3>().noalias() = q0 * T10_fej.linear();
  j_geo.du1_dx0.noalias() = du1_dp1 * dp_dx;

  // dp1_dr1 = [p1l']x
  dp_dx.leftCols<3>() = Hat3d(pt1l);
  // dp1_dt1 = -q0 * I
  dp_dx.rightCols<3>().noalias() = -q0 * Eigen::Matrix3d::Identity();
  j_geo.du1_dx1.noalias() = du1_dp1 * dp_dx;

  // dp1_dq0 = t10 - b
  Vector3d dp1_dq0 = T10_fej.translation();
  dp1_dq0.x() -= b;
  j_geo.du1_dq0.noalias() = du1_dp1 * dp1_dq0;

  return j_geo;
}

bool AdjustCost::UpdateHess(const Patch& patch0,
                            const FramePoint& point0,
                            const Patch& patch1,
                            const Vector3d& pt1l,
                            FrameHessian2& hess01,
                            int cam_ind) const noexcept {
  // Cost function is
  // r(k) = (I1[u'] - b1) - (I0[u] - b0) * e^(a1 - a0)
  const int ia = cam_ind + 1;  // affine ind
  const auto e_a1ma0 = eas[ia] / eas[0];
  const ArrayKd v0as = e_a1ma0 * (patch0.vals - bs[0]);
  const ArrayKd rs = (patch1.vals - bs[ia]) - v0as;
  const ArrayKd rs2 = rs.square();
  const ArrayKd gs2 = patch1.GradSqNorm();

  // Check for number of outliers
  if (IsWarpBad(gs2, rs2)) return false;

  //  const auto wi = static_cast<int>(point0.InfoMax()) + 1;
  const auto ws = CalcWeight(gs2, rs2);
  const auto It = patch1.gxys().eval();

  PatchHessian2 ph;
  ph.SetI(It, rs, ws);

  int aff = -1;
  if (cfg.affine) {
    aff = cam_ind * Dim::kAffine;

    Patch::Matrix2Kd A0t;
    const auto e_a1ma0_fej = eas_fej[ia] / eas_fej[0];
    const ArrayKd v0as_fej = e_a1ma0_fej * (patch0.vals - bs_fej[0]);
    // dr_da0 = e^(a1 - a0) * (v0 - b0) = v0a
    // dr_db0 = e^(a1 - a0)
    // A0t.row(0) = v0as_fej;
    // A0t.row(1).setConstant(e_a1ma0_fej);
    A0t.row(0).setZero();
    A0t.row(1).setOnes();

    Patch::Matrix2Kd A1t;
    // dr_da1 = -e^(a1 - a0) * (v0 - b0) = -v0a
    // dr_db1 = -1
    // A1t.row(0) = -v0as_fej;
    // A1t.row(1).setConstant(-1);
    A1t.row(0).setZero();
    A1t.row(1).setConstant(-1);

    ph.SetA(It, A0t, A1t, rs, ws);
  }

  const auto Js = CalcJacGeo(point0, pt1l, cam_ind);
  hess01.AddPatchHess(ph, Js.du1_dx0, Js.du1_dx1, aff);

  const Vector3i ijh{hess01.ii(), hess01.jj(), point0.hid};
  pblock->AddPatchHess(ph, Js.du1_dx0, Js.du1_dx1, Js.du1_dq0, ijh, aff);

  return true;
}

/// ============================================================================
std::string BundleAdjuster::Repr() const {
  return fmt::format("BundleAdjuster(cfg={})", cfg_.Repr());
}

AdjustStatus BundleAdjuster::Adjust(KeyframePtrSpan keyframes,
                                    const Camera& camera,
                                    int gsize) {
  return AdjustImpl(keyframes, camera, cfg_.solve, cfg_.cost, gsize);
}

AdjustStatus BundleAdjuster::AdjustImpl(KeyframePtrSpan keyframes,
                                        const Camera& camera,
                                        const DirectSolveCfg& solve_cfg,
                                        const DirectCostCfg& cost_cfg,
                                        int gsize) {
  AdjustStatus status{};

  const auto num_kfs = static_cast<int>(keyframes.size());
  CHECK_GE(num_kfs, 2) << "Need more than 2 keyframes for adjustment";
  VLOG(1) << "- num kfs: " << num_kfs;
  const auto num_points = PointsInfoGe(keyframes, DepthPoint::kMinInfo);
  VLOG(1) << "- num points: " << num_points;
  const auto num_levels = solve_cfg.GetNumLevels(keyframes.back()->levels());
  VLOG(1) << "- num levels: " << num_levels;
  const auto block_size = block_.MapFull(num_kfs, num_points);
  VLOG(1) << "- block size: " << block_size;
  const auto schur_size = schur_.MapFrames(num_kfs);
  VLOG(1) << "- schur size: " << schur_size;

  for (int level = num_levels - 1; level >= 0; --level) {
    VLOG(2) << "=== Level " << level;

    for (int iter = 0; iter < solve_cfg.max_iters; ++iter) {
      block_.ResetFull();

      BuildLevel(keyframes, camera, level, gsize);
      Solve(!cost_cfg.stereo, gsize);
      Update(keyframes, gsize);

      if (OnIterEnd(status,
                    level,
                    iter,
                    block_.num(),
                    block_.cost(),
                    block_.xp.squaredNorm() / num_kfs)) {
        break;
      }
    }  // iter
  }    // level

  // Update kf status info and linearization point
  for (auto* pkf : keyframes) {
    pkf->UpdateStatusInfo();
    pkf->UpdateLinearizationPoint();
  }

  status.num_kfs = num_kfs;
  status.num_points = num_points;
  status.num_levels = num_levels;
  status.max_iters = solve_cfg.max_iters * num_levels;
  return status;
}

void BundleAdjuster::Marginalize(KeyframePtrSpan keyframes,
                                 const Camera& camera,
                                 int kf_ind,
                                 int gsize) {
  // Pre-conditions
  // 1. must have at least 2 kfs
  const auto num_kfs = static_cast<int>(keyframes.size());
  CHECK_GE(kf_ind, 0);
  CHECK_LT(kf_ind, num_kfs);
  CHECK_GE(num_kfs, 2);

  VLOG(1) << "- Marg kf: " << kf_ind;
  // Only marginalize points with Ok info, discard rest
  const auto num_points = PointsInfoGe(keyframes, DepthPoint::kOkInfo);
  const auto block_size = block_.MapFull(num_kfs, num_points);
  const auto schur_size = schur_.MapFrames(num_kfs);

  VLOG(1) << "- block size: " << block_size;
  VLOG(1) << "- schur size: " << schur_size;

  // Construct hessian with residuals that depend only on the kf to be marged
  // and all points in it at full resolution
  block_.ResetFull();
  std::mutex mtx;
  BuildLevelKf(keyframes, camera, /*level*/ 0, kf_ind, mtx, gsize);
  VLOG(1) << "- block diag: " << block_.DiagSum();
  const auto& prange = pranges_.at(kf_ind);

  // 2. Marginalize all points in frame k, discard residual that would affect
  // the sparsity pattern (this was actually done in the previous step, where we
  // ignore Hessian blocks originated from points in frames other than k)
  // Schur will be modified so no need to reset
  block_.MargPointsRange(schur_, prange, gsize);
  VLOG(1) << "- points marged: " << schur_.n;

  // 3. If there exists a previous prior, we add it to the current schur
  if (prior_.Ok()) {
    schur_.AddPriorHess(prior_);
  }

  // 4. We then marginalize frame k to form a new prior
  const auto prior_size = prior_.MapFrames(num_kfs - 1);
  VLOG(1) << "- prior size: " << prior_size;
  schur_.MargFrame(prior_, kf_ind);
}

void BundleAdjuster::BuildLevel(KeyframePtrSpan keyframes,
                                const Camera& camera,
                                int level,
                                int gsize) {
  const auto camera_scaled = camera.AtLevel(level);
  const auto num_kfs = static_cast<int>(keyframes.size());

  // NOTE: each cost function will connect two frames and one point, so ideally
  // we could process each frame in parallel. This is because the point hessian
  // of each frame is disjoint so it's ok to modify it from multiple threads as
  // long as each thread is looking at a different frame (thus different set of
  // points). However, if there are more than two keyframes, then the returned
  // FrameHessian2 cannot be easily reduced since they have different blocks.
  // Therefore, we have to protect the FullBlockHessian with a mutex and update
  // it when we get a final FrameHessian2. We only need to lock a few times (n *
  // (n-1)), thus the mutex is not highly-contended, but the extra
  // parallelization improves cpu utilization and reduce runtime by ~20% so we
  // will keep it unless there is a problem later.

  std::mutex mtx;
  ParallelFor({0, num_kfs, gsize}, [&](int k0) {
    BuildLevelKf(keyframes, camera_scaled, level, k0, mtx, gsize);
  });
}

void BundleAdjuster::BuildLevelKf(KeyframePtrSpan keyframes,
                                  const Camera& camera,
                                  int level,
                                  int k0,
                                  std::mutex& mtx,
                                  int gsize) {
  //  const auto dinfo = 1.0 / cfg_.solve.max_iters / (level + 1.0);
  // This will make points converge a bit faster
  const auto dinfo =
      2.0 / (DepthPoint::kMaxInfo - DepthPoint::kOkInfo) / (level + 1.0);
  const auto num_kfs = static_cast<int>(keyframes.size());
  const auto& kf0 = GetKfAt(keyframes, k0);

  for (int k1 = 0; k1 < num_kfs; ++k1) {
    // Skip the case when looking at the same frame, note that this also
    // prevents us from projecting left to right in the same frame in
    // stereo version.
    if (k0 == k1) continue;

    const auto& kf1 = GetKfAt(keyframes, k1);
    const auto& points0 = kf0.points();
    const auto& patches0 = kf0.patches().at(level);
    const AdjustCost cost(
        level, camera, kf0, kf1, {k0, k1}, block_, cfg_.cost, dinfo);
    const auto hess01 = cost.Build(points0, patches0, gsize);
    // lock hess, this only gets called O(n^2) times
    std::scoped_lock lock{mtx};
    block_.AddFrameHess(hess01);
  }
}

void BundleAdjuster::Solve(bool fix_scale, int gsize) {
  CHECK(block_.Ok()) << block_.Repr();
  const auto num_kfs = block_.num_frames();
  CHECK_EQ(num_kfs, schur_.num_frames());

  // 1. Invert Hmm and make Hpp symmetric
  block_.Prepare();

  // 2. Marginalize all points
  block_.MargPointsAll(schur_, gsize);
  VLOG(3) << "--- marg points: " << schur_.num();
  VLOG(3) << "--- schur diag before prior: " << schur_.DiagSum();

  constexpr double large = 1e16;
  constexpr double medium = 1e8;
  constexpr double small = 1e4;

  // 3. Add prior or fix gauge manually
  if (prior_.Ok()) {
    // We have a prior, so we add it to schur
    schur_.AddPriorHess(prior_);
    // However we also add a small fix to the first frame pose just in case
    schur_.AddDiag(0, Dim::kPose, medium);
  } else {
    // We don't have a prior, this could either due to not enabling marg or when
    // window is not filled at the beginning of the sequence. We thus manually
    // fix the gauge of the first frame and/or scale
    schur_.AddDiag(0, Dim::kPose, large);

    // In mono mode we also need to fix translational scale
    if (fix_scale) {
      schur_.AddDiag(Dim::kFrame + 3, 3, large);
    }
  }

  // 4. Regularize affine parameters
  // We need to fix the left affine params of the first frame, and put a small
  // penalty on the rest of the affine parameters
  for (int i = 0; i < num_kfs; ++i) {
    const int ia = i * Dim::kFrame + Dim::kPose;
    if (i == 0) {
      schur_.AddDiag(ia, Dim::kAffine, large);
      schur_.AddDiag(ia + Dim::kAffine, Dim::kAffine, small);
    } else {
      schur_.AddDiag(ia, Dim::kAffine * 2, small);
    }
  }

  // 5. Solve for xp and xm
  block_.Solve(schur_);
}

void BundleAdjuster::Update(KeyframePtrSpan keyframes, int gsize) {
  ParallelFor({0, static_cast<int>(keyframes.size()), gsize}, [&](int k) {
    Keyframe& kf = *keyframes.at(k);
    kf.UpdateState(block_.XpAt(k));
    kf.UpdatePoints(block_.xm, gsize);
  });
}

size_t BundleAdjuster::Allocate(int max_frames, int max_points_per_frame) {
  // it is very unlikely we will use max_points_per_frame, thus we reserve
  // reduced size
  block_.ReserveFull(max_frames, max_points_per_frame * max_frames);
  schur_.ReserveData(max_frames);
  prior_.ReserveData(max_frames - 1);

  return (block_.capacity() + schur_.capacity() + prior_.capacity()) *
         sizeof(double);
}

}  // namespace sv::dsol

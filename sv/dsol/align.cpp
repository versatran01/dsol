#include "sv/dsol/align.h"

#include "sv/dsol/pixel.h"
#include "sv/util/logging.h"
#include "sv/util/tbb.h"

namespace sv::dsol {

using Vector2d = Eigen::Vector2d;

std::string FrameAligner::Repr() const {
  return fmt::format("FrameAligner(cfg={})", cfg_.Repr());
}

size_t FrameAligner::Allocate(size_t num_kfs, const cv::Size& grid_size) {
  if (points1_vec_.size() != num_kfs) {
    points1_vec_.resize(num_kfs, DepthPointGrid(grid_size));
  }
  return num_kfs * points1_vec_.size() * sizeof(DepthPoint);
}

AlignStatus FrameAligner::Align(KeyframePtrSpan keyframes,
                                const Camera& camera,
                                Frame& frame,
                                int gsize) {
  return AlignImpl(keyframes, camera, frame, cfg_.solve, cfg_.cost, gsize);
}

AlignStatus FrameAligner::AlignImpl(KeyframePtrSpan keyframes,
                                    const Camera& camera,
                                    Frame& frame,
                                    const DirectSolveCfg& solve_cfg,
                                    const DirectCostCfg& cost_cfg,
                                    int gsize) {
  AlignStatus status{};

  const auto num_kfs = static_cast<int>(keyframes.size());
  CHECK_GE(num_kfs, 1) << "Need more than 1 keyframe for alignment";
  VLOG(1) << "- num kfs: " << num_kfs;
  const auto num_bytes = Allocate(num_kfs, keyframes[0]->points().cvsize());
  VLOG(1) << "- num bytes: " << num_bytes;
  const auto num_points = PointsInfoGe(keyframes, DepthPoint::kOkInfo);
  VLOG(1) << "- num points: " << num_points;
  const auto num_levels = solve_cfg.GetNumLevels(frame.levels());
  VLOG(1) << "- num levels: " << num_levels;
  const auto dim_frame = cost_cfg.GetFrameDim();
  VLOG(1) << "- dim frame: " << dim_frame;

  Frame::Vector10d x;
  for (int level = num_levels - 1; level >= 0; --level) {
    VLOG(2) << "=== Level " << level;

    for (int iter = 0; iter < solve_cfg.max_iters; ++iter) {
      auto hess = BuildLevel(keyframes, camera, frame, level, gsize);
      const auto ok = hess.Solve(x, dim_frame);

      // Update frame parameters
      frame.UpdateState(x);

      // Update status and check for early stop
      if (OnIterEnd(
              status, level, iter, hess.num(), hess.cost(), x.squaredNorm())) {
        break;
      }
    }  // iter
  }    // level

  // Update kf track status
  for (size_t k = 0; k < keyframes.size(); ++k) {
    keyframes.at(k)->UpdateStatusTrack(points1_vec_.at(k));
  }

  status.num_kfs = num_kfs;
  status.num_points = num_points;
  status.num_levels = num_levels;
  status.max_iters = num_levels * solve_cfg.max_iters;
  return status;
}

FrameHessian1 FrameAligner::BuildLevel(KeyframePtrSpan keyframes,
                                       const Camera& camera,
                                       const Frame& frame,
                                       int level,
                                       int gsize) {
  const auto camera_scaled = camera.AtLevel(level);
  const auto num_kfs = static_cast<int>(keyframes.size());

  return ParallelReduce(
      {0, num_kfs, gsize},
      FrameHessian1{},
      [&](int k0, FrameHessian1& hess) {
        const auto& kf0 = GetKfAt(keyframes, k0);

        auto& points1 = points1_vec_.at(k0);
        const auto& points0 = kf0.points();
        const auto& patches0 = kf0.patches().at(level);
        const AlignCost cost(level, camera_scaled, kf0, frame, cfg_.cost);
        hess += cost.Build(points0, patches0, points1, gsize);
      },
      std::plus<>{});
}

cv::Mat FrameAligner::CalcCellIdepth(int cell_size) const {
  if (points1_vec_.empty()) return {};

  static cv::Mat idepth;
  idepth.create(points1_vec_.front().cvsize(), CV_64FC2);
  idepth.setTo(0);

  for (const auto& points1 : points1_vec_) {
    for (const auto& point1 : points1) {
      // We only check for info ok because we reset all points to invalid during
      // the alignment process and thus guarantees that InfoOK() implies pixel
      // and depth ok
      // TODO (dsol): Ok or Max?
      if (!point1.InfoOk()) continue;
      CHECK(point1.PixelOk());
      CHECK(point1.DepthOk());

      // compute which cell this point falls into
      const int r = static_cast<int>(point1.px().y) / cell_size;
      const int c = static_cast<int>(point1.px().x) / cell_size;
      //      if (IsPixOut(idepth, cv::Point(r, c))) continue;
      if (r < 0 || r >= idepth.rows || c < 0 || c >= idepth.cols) continue;

      auto& cell = idepth.at<cv::Vec2d>(r, c);
      cell[0] += point1.idepth();  // sum of idepth
      cell[1] += 1;                // num of idepth
    }
  }

  return idepth;
}

/// ============================================================================
AlignCost::AlignCost(int level,
                     const Camera& camera_scaled,
                     const Keyframe& keyframe,
                     const Frame& frame,
                     const DirectCostCfg& cost_cfg)
    : DirectCost(level, camera_scaled, frame, cost_cfg) {
  const auto& st0 = keyframe.state();
  const auto& st1 = frame.state();
  ExtractState(st0, st1, T10, eas, bs);
}

FrameHessian1 AlignCost::Build(const FramePointGrid& points0,
                               const PatchGrid& patches0,
                               DepthPointGrid& points1,
                               int gsize) const {
  return ParallelReduce(
      {0, points0.rows(), gsize},
      FrameHessian1{},
      [&](int gr, FrameHessian1& hess) {
        for (int gc = 0; gc < points0.cols(); ++gc) {
          auto& point1 = points1.at(gr, gc);
          point1 = {};  // reset

          const auto& point0 = points0.at(gr, gc);
          if (point0.HidBad()) continue;

          // Note that even with a good point we might have a bad patch in low
          // res image due to down scaling. So we need to check if this patch
          // is ok to use.
          const auto& patch0 = patches0.at(gr, gc);
          if (patch0.Bad()) continue;

          point1 = WarpPatch(patch0, point0, hess);
        }  // gc
      },   // gr
      std::plus<>{});
}

DepthPoint AlignCost::WarpPatch(const Patch& patch0,
                                const FramePoint& point0,
                                FrameHessian1& hess) const noexcept {
  // First construct all uvs of this patch
  const auto uv0 = ScaleUv(point0.uv(), camera.scale());
  const auto uv0s = (Patch::offsets().colwise() + uv0).eval();

  // VERY IMPORTANT NOTE: The normal warping that uses depth is
  //
  // p1  = R10 * n0 / q0 + t10 -> n1 = Proj(p1)
  //
  // However, due to the projection afterward, it is equivalent to
  //
  // p1' = R10 * n0 + q0 * t10 -> n1' = Proj(p1')
  //
  // This can be thought of as scaling everything by q0, which simplifies
  // Jacobian computation

  // Then do a batch warp for all pixels in patch (q is inv depth, z is depth, d
  // is disparity)
  const auto q0 = point0.idepth();
  const auto nh0s = camera.Backward(uv0s);
  const auto pt1s = TransformScaled(T10, nh0s, q0);

  Patch::Point2dArray px1s{};
  Eigen::Map<Matrix2Kd> uv1s(&px1s[0].x);
  uv1s = camera.Forward(pt1s);

  DepthPoint point1{};
  point1.SetPix(px1s.front());

  // Thus the depth of the point in frame1 can be computed by
  // p1 = p1' / q0
  // z1 = z1' / q0
  // and the inverse depth in frame1 is
  // q1 = 1 / z1 = q0 / z1'
  const auto z1 = pt1s.col(0).z() / q0;  // z1' has no scale
  point1.idepth_ = 1 / z1;               // z1 could be negatvie
  point1.info_ = DepthPoint::kMinInfo;   // denotes OOB
  //  point1.SetIdepthInfo(1 / z1, 0);       // 0 denotes OOB

  // Skip points that project too close or to the back of the camera
  // z1 cloud be nan since z1' and q0 could both be 0
  if (!(z1 > cfg.min_depth)) return point1;

  // Stores warped patch value
  Patch patch1;

  // If no OOB, we update hessian for this patch
  if (!Patch::IsAnyOut(gray1l, px1s)) {
    // Extract pixel intensity from left image
    patch1.ExtractIntensity(gray1l, px1s);

    // Left image cam_ind = 0
    const auto good = UpdateHess(patch0, point0, patch1, hess, 0);
    point1.info_ = good ? DepthPoint::kOkInfo : DepthPoint::kBadInfo;
  }

  if (cfg.stereo) {
    ApplyDisp(uv1s, pt1s.row(2).array(), q0);

    // Check for OOB
    if (!Patch::IsAnyOut(gray1r, px1s)) {
      // Extract pixel intensity from right image, can reuse storage
      patch1.ExtractIntensity(gray1r, px1s);

      // Right image cam_ind = 1
      UpdateHess(patch0, point0, patch1, hess, 1);
    }
  }

  return point1;
}

AlignCost::JacGeo AlignCost::CalcJacGeo(const FramePoint& point0,
                                        int cam_ind) const noexcept {
  JacGeo du_dx;

  //  if (cam_ind == 0) {
  //    du_dx.noalias() = camera.fxy().matrix().asDiagonal() * point0.dn_dx;
  //    return du_dx;
  //  }

  // n = Proj(p) = [x/z, y/z]^T
  // p = [x, y, 1]^T
  // dn_dp = [1/z,  0, -x/z^2] = [1, 0, -x]
  //         [0,  1/z, -y/z^2]   [0, 1, -y]

  const auto& nc0 = point0.nc;
  FramePoint::Matrix23d du_dp;
  du_dp << 1, 0, -nc0.x(), 0, 1, -nc0.y();

  // u = K * n, ux = fx * x + cx, uy = fy * y + cy
  // du_dn = [fx, 0]
  //         [0, fy]
  // du_dp = du_dn * dn_dp

  du_dp.array().colwise() *= camera.fxy();

  // du_dx = du_dp * dp_dx
  // dp_dx = [dp_dr, dp_dt]
  // p  = dR * n0 / q0 + dt
  // p' = dR * n       + dt * q0
  // dp_dr = -[n]x
  // dp_dt = q0 * I
  // TODO (dsol): add derivation of right frame

  const auto q0 = point0.idepth();
  const double bq0 = q0 * camera.baseline() * cam_ind;
  du_dx.leftCols<3>().noalias() = -du_dp * Hat3d(nc0.x() + bq0, nc0.y(), 1.0);
  du_dx.rightCols<3>().noalias() = du_dp * q0;

  return du_dx;
}

bool AlignCost::UpdateHess(const Patch& patch0,
                           const FramePoint& point0,
                           const Patch& patch1,
                           FrameHessian1& hess,
                           int cam_ind) const noexcept {
  // Cost function is
  // r = (I0[u] - b0) - e^(a0 - a1) * (I1[u'] - b1)
  const int ia = cam_ind + 1;  // ia retrieves affine param
  const auto e_a0ma1 = eas[0] / eas[ia];
  const ArrayKd v1as = e_a0ma1 * (patch1.vals - bs[ia]);
  const ArrayKd rs = (patch0.vals - bs[0]) - v1as;
  const ArrayKd rs2 = rs.square();
  const ArrayKd gs2 = patch0.GradSqNorm();

  if (IsWarpBad(gs2, rs2)) return false;

  // wi = 2 if info == max else 1
  //  const auto wi = static_cast<int>(point0.InfoMax()) + 1;
  const auto ws = CalcWeight(gs2, rs2);
  const auto It = patch0.gxys().eval();

  PatchHessian1 ph;
  ph.SetI(It, rs, ws);

  int aff = -1;
  if (cfg.affine) {
    aff = cam_ind * Dim::kAffine;

    Patch::Matrix2Kd At;
    // dr_da1 = e^(a0 - a1) * (v1 - b1) = v1a
    // dr_db1 = e^(a0 - a1)
    // At.row(0) = v1as;
    // At.row(1).setConstant(e_a0ma1);
    At.row(0).setZero();
    At.row(1).setOnes();
    ph.SetA(It, At, rs, ws);
  }

  const auto dn_dx = CalcJacGeo(point0, cam_ind);
  hess.AddPatchHess(ph, dn_dx, aff);

  return true;
}

}  // namespace sv::dsol

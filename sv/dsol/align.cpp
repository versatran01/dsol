#include "sv/dsol/align.h"

#include "sv/dsol/pixel.h"
#include "sv/dsol/solve.h"
#include "sv/util/logging.h"
#include "sv/util/tbb.h"

namespace sv::dsol {

using Vector2d = Eigen::Vector2d;

void Proj2Mask(const DepthPointGrid& points1,
               const cv::Size& cell_size,
               cv::Mat& mask) {
  for (const auto& point1 : points1) {
    // We only check for info ok because we reset all points to invalid during
    // align and thus guarantees that InfoOK() implies pixel and depth ok
    if (!point1.InfoOk()) continue;
    CHECK(point1.PixelOk());
    CHECK(point1.DepthOk());

    // compute which cell this point falls into
    const int r = static_cast<int>(point1.px().y) / cell_size.height;
    const int c = static_cast<int>(point1.px().x) / cell_size.width;
    if (IsPixOut(mask, cv::Point(c, r))) continue;

    mask.at<uint8_t>(r, c) += 1;
  }
}

/// @brief Accumulate projection results to idepths grid
void Proj2Idepth(const DepthPointGrid& points1,
                 const cv::Size& cell_size,
                 cv::Mat& idepth) {
  for (const auto& point1 : points1) {
    // We only check for info ok because we reset all points to invalid during
    // align and thus guarantees that InfoOK() implies pixel and depth ok
    if (!point1.InfoOk()) continue;
    CHECK(point1.PixelOk());
    CHECK(point1.DepthOk());

    // compute which cell this point falls into
    const int r = static_cast<int>(point1.px().y) / cell_size.height;
    const int c = static_cast<int>(point1.px().x) / cell_size.width;
    if (IsPixOut(idepth, cv::Point(c, r))) continue;
    auto& cell = idepth.at<cv::Vec2d>(r, c);
    cell[0] += point1.idepth();  // idepth sum
    ++cell[1];                   // count
  }
}

std::string FrameAligner::Repr() const {
  return fmt::format("FrameAligner(cfg={})", cfg_.Repr());
}

void FrameAligner::Reset() {
  points1_vec_.clear();
  num_tracks_.clear();
  idepth_.setTo(0);
}

size_t FrameAligner::Allocate(size_t num_kfs, const cv::Size& grid_size) {
  if (points1_vec_.size() != num_kfs) {
    points1_vec_.resize(num_kfs, DepthPointGrid(grid_size));
  }

  if (idepth_.empty()) {
    idepth_ = cv::Mat::zeros(grid_size, CV_64FC2);
  }

  return num_kfs * points1_vec_.size() * sizeof(DepthPoint) +
         idepth_.total() * idepth_.elemSize();
}

AlignStatus FrameAligner::Align(KeyframePtrSpan keyframes,
                                const Camera& camera,
                                Frame& frame,
                                int gsize) {
  AlignStatus status{};

  const auto num_kfs = static_cast<int>(keyframes.size());
  CHECK_GE(num_kfs, 1) << "Need more than 1 keyframe for alignment";
  VLOG(1) << "- num kfs: " << num_kfs;
  const auto num_bytes = Allocate(num_kfs, keyframes[0]->points().cvsize());
  VLOG(1) << "- num bytes: " << num_bytes;
  const auto init_level = cfg_.optm.GetInitLevel(frame.levels());
  VLOG(1) << "- init level: " << init_level;

  for (int level = init_level; level >= 0; --level) {
    const auto status_l = AlignLevel(keyframes, camera, frame, level, gsize);
    status.Accumulate(status_l);
  }

  UpdateTrackAndIdepth(keyframes, frame);

  status.num_kfs = num_kfs;
  status.num_points = pranges_.back().end;
  return status;
}

AlignStatus FrameAligner::Align2(KeyframePtrSpan keyframes,
                                 const Camera& camera,
                                 Frame& frame,
                                 int gsize) {
  AlignStatus status{};

  const auto num_kfs = keyframes.size();
  CHECK_GE(num_kfs, 1) << "Need more than 1 keyframe for alignment";
  VLOG(1) << "- num kfs: " << num_kfs;
  const auto num_bytes = Allocate(num_kfs, keyframes[0]->points().cvsize());
  VLOG(1) << "- num bytes: " << num_bytes;
  const auto init_level = cfg_.optm.GetInitLevel(frame.levels());
  VLOG(1) << "- init level: " << init_level;

  // Get a copy of the old state in case align failedj
  const FrameState init_state = frame.state();

  // Stage 1: search for the highest level that converges.
  int top_level = init_level;
  // top_level is where we start searching for convergence, we start from the
  // current level and see if we can align frame. If the result is converged,
  // then we start normal mutil-scale alignment. If not, we go one level up (to
  // a lower resolution) until we reach the top. At that point, the entire
  // alignment is deemed not converged and we should reinitialize.
  for (; top_level < frame.levels(); ++top_level) {
    status = AlignLevel(keyframes, camera, frame, top_level, gsize);
    if (status.converged) break;

    // Revert frame state and restart
    frame.SetState(init_state);
  }

  // Update status
  status.num_kfs = static_cast<int>(num_kfs);
  status.num_points = pranges_.back().end;

  // Once we've exhausted all lower levels without convergence, we will stop
  // right here and mark the entire alignment failed.
  if (!status.converged) {
    return status;
  }

  // Stage 2: If we found a converged level, we can start from there
  for (int level = top_level - 1; level >= 0; --level) {
    const auto status_l = AlignLevel(keyframes, camera, frame, level, gsize);
    status.Accumulate(status_l);
  }

  UpdateTrackAndIdepth(keyframes, frame);
  return status;
}

AlignStatus FrameAligner::AlignLevel(KeyframePtrSpan keyframes,
                                     const Camera& camera,
                                     Frame& frame,
                                     int level,
                                     int gsize) {
  VLOG(2) << fmt::format("=== Level {} ===", level);

  const auto dim_frame = cfg_.cost.GetFrameDim();
  VLOG(2) << "-- dim frame: " << dim_frame;

  const auto oob_count = std::clamp(level, 2, 5);
  const auto num_points = PrepPoints(keyframes, oob_count);
  VLOG(2) << "-- num points: " << num_points;

  AlignStatus status;
  status.num_kfs = keyframes.size();
  status.num_levels = 1;
  status.num_points = num_points;

  double xs_prev = 1e10;
  const auto xs_stop = cfg_.optm.max_xs * std::pow(2.0, level);
  // adaptive iters, more iters at higher res
  const auto max_iters = cfg_.optm.max_iters;

  for (int iter = 0; iter < max_iters; ++iter) {
    const auto hess = BuildLevel(keyframes, camera, frame, level, gsize);

    // Check if we have enough costs to constrain the solution
    if (hess.num_costs() < dim_frame * status.num_kfs * 2) {
      VLOG(2) << LogIter({level, frame.levels()},
                         {iter, max_iters},
                         {hess.num_costs(), hess.cost()});
      status.num_levels = 0;
      status.converged = false;
      VLOG(1) << fmt::format("=== Level {} not enough costs {}", level, status);
      return status;
    }

    const auto xs_max = Solve(hess, dim_frame);

    // This is a hacky LM implementation without re-evaluation costs twice,
    // because it's too expensive. Instead we just check if the l-inf norm of
    // the normalized update (x) is smaller than the one from the previous iter.
    // If yes, then we accept this iter. If no, we still move along the same
    // direction, but with a reduced step size.
    frame.UpdateState(x_);
    // if (xs_max <= xs_prev * 2) {
    //   frame.UpdateState(x_);
    //   ++status.good_iters;
    // } else {
    //   frame.UpdateState(x_ * 0.5);
    // }
    xs_prev = xs_max;

    VLOG(2) << LogIter({level, frame.levels()},
                       {iter, max_iters},
                       {hess.num_costs(), hess.cost()},
                       {xs_max, xs_stop},
                       x_.squaredNorm());

    ++status.num_iters;
    status.cost = hess.cost();
    status.num_costs = hess.num_costs();

    // converged
    if (xs_max < xs_stop && iter > 1) {
      status.converged = true;
      break;
    }
  }

  VLOG(1) << LogConverge(level, status);
  return status;
}

void FrameAligner::UpdateTrackAndIdepth(KeyframePtrSpan keyframes,
                                        const Frame& frame) {
  const auto img_size = frame.cvsize();
  const cv::Size cell_size(img_size.width / idepth_.cols,
                           img_size.height / idepth_.rows);
  idepth_.setTo(0);
  num_tracks_.resize(keyframes.size());

  // Update tracks and idepth grid
  for (size_t k = 0; k < keyframes.size(); ++k) {
    const auto& points1 = points1_vec_.at(k);
    Proj2Idepth(points1, cell_size, idepth_);
    num_tracks_.at(k) = static_cast<int>(std::count_if(
        points1.cbegin(), points1.cend(), [](const DepthPoint& p) {
          return p.InfoOk();
        }));
  }
}

double FrameAligner::Solve(const FrameHessian1& hess, int dim, double lambda) {
  CHECK(hess.Ok()) << "num: " << hess.num_costs();

  x_.setZero();
  y_.setZero();

  SolveCholeskyScaled(hess.H.topLeftCorner(dim, dim),
                      hess.b.head(dim),
                      x_.head(dim),
                      y_.head(dim));
  return y_.head<Dim::kPose>().array().abs().maxCoeff();
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
      [&](int k, FrameHessian1& hess) {
        auto& points1 = points1_vec_.at(k);
        const auto& kf0 = GetKfAt(keyframes, k);
        const auto& points0 = kf0.points();
        const auto& patches0 = kf0.patches().at(level);

        const AlignCost cost(level, camera_scaled, kf0, frame, cfg_.cost);
        hess += cost.Build(points0, patches0, points1, gsize);
      },
      std::plus<>{});
}

int FrameAligner::PrepPoints(KeyframePtrSpan keyframes, int count) {
  const auto num_kfs = keyframes.size();
  pranges_.resize(num_kfs);

  int hid = 0;
  // Assign a hessian index to each good point
  for (size_t k = 0; k < num_kfs; ++k) {
    auto& kf = GetKfAt(keyframes, k);
    auto& prg = pranges_.at(k);

    prg.start = hid;
    for (auto& point : kf.points()) {
      if (point.InfoOk()) {
        CHECK(point.PixelOk());
        CHECK(point.DepthOk());
        ++hid;
        point.hid = count;
      } else {
        point.hid = FramePoint::kBadHid;
      }
    }
    prg.end = hid;
  }  // k

  CHECK_GT(hid, 0);
  return hid;
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
  points1.reset();  // reset for new round

  return ParallelReduce(
      {0, points0.rows(), gsize},
      FrameHessian1{},
      [&](int gr, FrameHessian1& hess) {
        for (int gc = 0; gc < points0.cols(); ++gc) {
          auto& point1 = points1.at(gr, gc);

          const auto& point0 = points0.at(gr, gc);
          if (point0.HidBad()) continue;

          // Note that even with a good point we might have a bad patch in low
          // res image due to down scaling. So we need to check if this patch
          // is ok to use.
          const auto& patch0 = patches0.at(gr, gc);
          if (patch0.Bad()) continue;

          point1 = WarpPatch(patch0, point0, hess);
          // point1.info shows the warp result
          // MinInfo -> OOB or depth < 0
          // OkInfo  -> Inlier
          // BadInfo -> Outlier
          // If point1 is OOB, we decrease its hid. If a point is consistently
          // OOB, then after a while its hid will be < 0 and it will no longer
          // be considered during alignment. This is a performance
          // optimization
          if (point1.info() == DepthPoint::kMinInfo) --point0.hid;
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
  
  // A much better explanation can be found at
  // https://github.com/symforce-org/symforce/blob/d756d4d2c5dba37f3ae36ebc3fa40ccba8ace5e6/symforce/cam/posed_camera.py#L90
  
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

  // Then do a batch warp for all pixels in patch (q is inv depth, z is depth,
  // d is disparity)
  const auto q0 = point0.idepth();
  const auto nh0s = camera.Backward(uv0s);
  const auto pt1s = TransformScaled(T10, nh0s, q0);

  Patch::Point2dArray px1s{};
  Eigen::Map<Matrix2Kd> uv1s(&px1s[0].x);
  uv1s = camera.Forward(pt1s);

  DepthPoint point1{};
  point1.SetPix(ScalePix(px1s.front(), 1.0 / camera.scale()));

  // Thus the depth of the point in frame1 can be computed by
  // p1 = p1' / q0
  // z1 = z1' / q0
  // and the inverse depth in frame1 is
  // q1 = 1 / z1 = q0 / z1'
  const auto z1 = pt1s.col(0).z() / q0;  // z1' has no scale
  point1.idepth_ = 1 / z1;               // z1 could be negatvie
  point1.info_ = DepthPoint::kMinInfo;   // denotes OOB

  // Skip points that project too close or to the back of the camera
  // z1 cloud be nan since z1' and q0 could both be 0
  if (!(z1 > cfg.min_depth)) return point1;

  // Stores warped patch value
  Patch patch1;

  // If no OOB, we update hessian for this patch
  if (!Patch::IsAnyOut(gray1l, px1s, 1)) {
    // Extract pixel intensity from left image
    patch1.ExtractIntensity(gray1l, px1s);

    // Left image cam_ind = 0
    const auto ok = UpdateHess(patch0, point0, patch1, hess, 0);
    point1.info_ = ok ? DepthPoint::kOkInfo : DepthPoint::kBadInfo;
  }

  if (cfg.stereo) {
    ApplyDisp(uv1s, q0 / pt1s.row(2).array());

    // Check for OOB
    if (!Patch::IsAnyOut(gray1r, px1s, 1)) {
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

  // Check for number of outliers
  //  if (IsWarpBad(gs2, rs2)) return false;
  const auto no = GetNumOutliers(gs2, rs2);
  if (no > cfg.max_outliers) return false;
  //  const auto wi = FastPoint5Pow(no);

  const auto ws = CalcWeight(gs2, rs2);
  const auto It = patch0.gxys().eval();

  PatchHessian1 ph;
  ph.SetI(It, rs, ws);

  int affi = -1;
  if (cfg.affine) {
    affi = cam_ind * Dim::kAffine;

    Patch::Matrix2Kd At;
    // dr_da1 = e^(a0 - a1) * (v1 - b1) = v1a
    // dr_db1 = e^(a0 - a1)
    At.row(0) = v1as;
    At.row(1).setConstant(e_a0ma1);
    // At.row(0).setZero();
    // At.row(1).setOnes();
    ph.SetA(It, At, rs, ws);
  }

  const auto dn_dx = CalcJacGeo(point0, cam_ind);
  hess.AddPatchHess(ph, dn_dx, affi);

  return true;
}

int CountIdepths(const cv::Mat& idepths) {
  int n = 0;
  for (int r = 0; r < idepths.rows; ++r) {
    for (int c = 0; c < idepths.cols; ++c) {
      const auto& cell = idepths.at<cv::Vec2d>(r, c);
      n += cell[1] > 0;
    }
  }
  return n;
}

}  // namespace sv::dsol

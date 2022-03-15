#include "sv/dsol/direct.h"

#include "sv/util/logging.h"

namespace sv::dsol {

void DirectSolveCfg::Check() const {
  CHECK_GT(max_iters, 0);
  CHECK_GE(max_levels, 0);
  CHECK_GE(rel_change, 0);
}

std::string DirectSolveCfg::Repr() const {
  return fmt::format(
      "DirectSolveCfg(max_iters={}, max_levels={}, rel_change={})",
      max_iters,
      max_levels,
      rel_change);
}

void DirectCostCfg::Check() const {
  CHECK_GT(c2, 0);
  CHECK_GT(dof, 0);
  CHECK_GE(max_outliers, 0);
  CHECK_GE(grad_factor, 1.0);
  CHECK_GT(min_depth, 0);
}

std::string DirectCostCfg::Repr() const {
  return fmt::format(
      "DirectCostCfg(affine={}, stereo={}, c2={}, dof={}, max_outliers={}, "
      "grad_factor={}, min_depth={})",
      affine,
      stereo,
      c2,
      dof,
      max_outliers,
      grad_factor,
      min_depth);
}

void DirectCfg::Check() const {
  solve.Check();
  cost.Check();
}

std::string DirectCfg::Repr() const {
  return fmt::format("DirectCfg({}, {})", solve.Repr(), cost.Repr());
}

/// ============================================================================
std::string DirectStatus::Repr() const {
  return fmt::format(
      "DirectStatus(num_kfs={}, num_points={}, num_levels={}, num_iters={}/{}, "
      "num_costs={}, last_cost={:.4e})",
      num_kfs,
      num_points,
      num_levels,
      num_iters,
      max_iters,
      num_costs,
      last_cost);
}

/// ============================================================================
DirectCost::DirectCost(int level,
                       const Camera& camera_scaled,
                       const Frame& frame1,
                       const DirectCostCfg& cost_cfg)
    : camera{camera_scaled}, gray1l{frame1.grays_l().at(level)}, cfg{cost_cfg} {
  CHECK(!gray1l.empty());
  // This is to make sure scale of camera matches scale of image pyramid
  CHECK_EQ(camera.width(), gray1l.cols);
  CHECK_EQ(camera.height(), gray1l.rows);

  if (cfg.stereo) {
    CHECK(camera.is_stereo()) << "cfg stereo but camera is mono";
    CHECK(frame1.is_stereo()) << "cfg stereo but frame1 is mono";
    gray1r = frame1.grays_r().at(level);
    CHECK(!gray1r.empty());
  }
}

void DirectCost::ExtractState(const FrameState& state0,
                              const FrameState& state1,
                              Eigen::Isometry3d& T_1_0,
                              Eigen::Array3d& eas,
                              Eigen::Array3d& bs) {
  // T_1_0 = T_1_w * T_w_0 = T_w_1^-1 * T_w_0
  T_1_0 = (state1.T_w_cl.inverse() * state0.T_w_cl).matrix();

  // We store affine parameters in the order of (0, 1l, 1r), so that we could
  // use camera index + 1 to retrieve the corresponding affine param
  // so cam_ind = 0 will get 1l and cam_ind = 1 will get 1r
  eas[0] = state0.affine_l.a();
  eas[1] = state1.affine_l.a();
  eas[2] = state1.affine_r.a();
  eas = eas.exp();

  bs[0] = state0.affine_l.b();
  bs[1] = state1.affine_l.b();
  bs[2] = state1.affine_r.b();
}

/// ============================================================================
bool DirectMethod::OnIterEnd(DirectStatus& status,
                             int level,
                             int iter,
                             int num,
                             double cost,
                             double dx2) const noexcept {
  const auto prev_mean_cost = status.last_cost / status.num_costs;
  const auto curr_mean_cost = cost / num;
  const auto rel_change = (curr_mean_cost - prev_mean_cost) / prev_mean_cost;
  const auto rel_change_max = cfg_.solve.rel_change * (level / 2.0 + 1);
  const auto rel_ok = std::abs(rel_change) < rel_change_max;
  const auto dx_ok = dx2 < status.last_dx2;

  const auto early_stop = rel_ok && dx_ok;

  ++status.num_iters;
  status.num_costs = num;
  status.last_cost = cost;
  status.last_dx2 = dx2;

  VLOG(2) << fmt::format(
      "-- [L {} I {}]: dx2={:.4e}, cost={:.4e}, num={}, mean={:.4f}, "
      "rel={:+.4f}/{:.4f}, stop={}",
      level,
      iter,
      dx2,
      cost,
      num,
      curr_mean_cost,
      rel_change,
      rel_change_max,
      early_stop);

  return early_stop;
}

int DirectMethod::PointsInfoGe(KeyframePtrSpan keyframes, double min_info) {
  const auto num_kfs = static_cast<int>(keyframes.size());
  pranges_.resize(num_kfs);

  int hid = 0;
  // Assign a hessian index to each good point
  for (int k = 0; k < num_kfs; ++k) {
    auto& kf = GetKfAt(keyframes, k);

    // We need to remember the start and end hid of each frame
    auto& hrg = pranges_.at(k);
    hrg.start = hid;

    // hid determines whether a point will be used in this round of adjustment
    // we will update point.info during the process, but hid remains the same,
    // so even if info goes < 0, as long as hid is valid, this point will still
    // participate in this round of bundle adjustment
    for (auto& point : kf.points()) {
      if (point.info() >= min_info) {
        CHECK(point.PixelOk());
        CHECK(point.DepthOk());
        point.hid = hid++;
      } else {
        point.hid = FramePoint::kBadHid;
      }
    }
    hrg.end = hid;
    VLOG(1) << fmt::format("-- kf {} has {} points, [{:>4}, {:>4})",
                           k,
                           hrg.size(),
                           hrg.start,
                           hrg.end);
  }  // k

  CHECK_GT(hid, 0);
  return hid;
}

}  // namespace sv::dsol

#include "sv/dsol/direct.h"

#include "sv/util/logging.h"

namespace sv::dsol {

namespace {
constexpr auto color_ok = LogColor::kBrightGreen;
constexpr auto color_bad = LogColor::kBrightRed;
}  // namespace

void DirectOptmCfg::Check() const {
  CHECK_GE(init_level, -2);
  CHECK_GT(max_iters, 0);
  CHECK_GE(max_xs, 0);
}

std::string DirectOptmCfg::Repr() const {
  return fmt::format("DirectOptmCfg(init_level={}, max_iters={}, max_xs={})",
                     init_level,
                     max_iters,
                     max_xs);
}

void DirectCostCfg::Check() const {
  CHECK_GT(c2, 0);
  CHECK_GT(dof, 0);
  CHECK_GE(max_outliers, 0);
  CHECK_LT(max_outliers, 3);
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
  optm.Check();
  cost.Check();
}

std::string DirectCfg::Repr() const {
  return fmt::format("{}, {}", optm.Repr(), cost.Repr());
}

/// ============================================================================
std::string DirectStatus::Repr() const {
  return fmt::format(
      "DirectStatus(num_kfs={}, num_points={}, num_levels={}, "
      "num_iters={}, num_costs={}, cost={:.2e}, converged={})",
      num_kfs,
      num_points,
      num_levels,
      num_iters,
      num_costs,
      cost,
      converged);
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
std::string DirectMethod::LogIter(const std::pair<int, int>& level,
                                  const std::pair<int, int>& iter,
                                  const std::pair<int, double>& num_cost,
                                  const std::pair<double, double>& xs,
                                  double x2) {
  return fmt::format(
      "-- [L {}/{} I {}/{}]: num={}, cost={:.2e}, xs={:.3f}/{:.3f}, "
      "x2={:.2e}",
      level.first,
      level.second,
      iter.first,
      iter.second,
      num_cost.first,
      num_cost.second,
      xs.first,
      xs.second,
      x2);
}

std::string DirectMethod::LogIter(const std::pair<int, int>& level,
                                  const std::pair<int, int>& iter,
                                  const std::pair<int, double>& num_cost) {
  return fmt::format("-- [L {}/{} I {}/{}]: num={}, cost={:.2e}",
                     level.first,
                     level.second,
                     iter.first,
                     iter.second,
                     num_cost.first,
                     num_cost.second);
}

std::string DirectMethod::LogConverge(int level, const DirectStatus& status) {
  if (status.converged) {
    return fmt::format(color_ok, "=== Level {} converged {}", level, status);
  }
  return fmt::format(color_bad, "=== Level {} diverged {}", level, status);
}

}  // namespace sv::dsol

#pragma once

#include "sv/dsol/frame.h"

namespace sv::dsol {

/// @brief Direct method optimization config
struct DirectOptmCfg {
  int init_level{0};   // init level to start optimzation
  int max_iters{8};    // num max iters per level
  double max_xs{0.1};  // max change in normalized change to stop early

  void Check() const;
  std::string Repr() const;

  /// @brief Get actual init level
  int GetInitLevel(int num_levels) const noexcept {
    const int max_level = num_levels - 1;
    return init_level <= 0 ? max_level + init_level
                           : std::min(init_level, max_level);
  }
};

/// @brief Direct method cost config
struct DirectCostCfg {
  bool affine{false};       // use affine brightness model
  bool stereo{false};       // use stereo images
  int c2{2};                // gradient weight c^2, w_g = c^2 / (c^2 + dI^2)
  int dof{4};               // dof for t-dist, w_r = (dof + 1) / (dof + r^2)
  int max_outliers{1};      // max outliers in each patch
  double grad_factor{1.5};  // r^2 > grad_factor * g^2 is outlier
  double min_depth{0.2};    // min depth to skip when project

  void Check() const;
  std::string Repr() const;

  /// @brief Get actual frame dim
  int GetFrameDim() const noexcept {
    return Dim::kPose + Dim::kAffine * affine * (stereo + 1);
  }
};

/// @brief Direct method config
struct DirectCfg {
  DirectOptmCfg optm;
  DirectCostCfg cost;

  void Check() const;
  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const DirectCfg& rhs) {
    return os << rhs.Repr();
  }
};

/// @brief Direct method status
struct DirectStatus {
  int num_kfs{};          // num kfs used
  int num_points{};       // num points used
  int num_levels{};       // num levels used
  int num_iters{};        // num iters run
  int good_iters{};       // good iters
  int num_costs{};        // num costs
  double cost{};          // total cost
  bool converged{false};  // whether convergence reached

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const DirectStatus& rhs) {
    return os << rhs.Repr();
  }

  void Accumulate(const DirectStatus& status) noexcept {
    num_levels += status.num_levels;
    good_iters += status.good_iters;
    num_iters += status.num_iters;
    num_costs = status.num_costs;
    cost = status.cost;
    converged = status.converged;
  }
};

/// @brief Direct method cost
struct DirectCost {
  using ArrayKd = Patch::ArrayKd;
  using Matrix2Kd = Patch::Matrix2Kd;

  DirectCost() = default;
  DirectCost(int level,
             const Camera& camera_scaled,
             const Frame& frame1,
             const DirectCostCfg& cost_cfg);

  /// @brief Calculate weight for each cost
  /// @param g2 is squared norm of image gradient
  /// @param r2 is squared norm of residual
  ArrayKd CalcWeight(const ArrayKd& g2,
                     const ArrayKd& r2,
                     double wi = 1.0) const noexcept {
    // return wg * (cfg.dof + 1) / (cfg.dof + r2 * wg);
    // return wi * (cfg.dof + 1) / (cfg.dof / wg + r2);
    // const auto num = wi * cfg.c2 * (cfg.dof + 1);
    // return num / (cfg.dof * (cfg.c2 + g2) + r2);
    const ArrayKd wg = cfg.c2 / (cfg.c2 + g2);
    return wg * (wi * (cfg.dof + 1)) / (cfg.dof + r2 * wg);
  }

  /// @brief Check if this warp is ok or not
  /// @details If residual bigger than gradient * grad_factor
  int GetNumOutliers(const ArrayKd& g2, const ArrayKd& r2) const noexcept {
    return static_cast<int>((r2 > (cfg.grad_factor * g2)).count());
  }
  bool IsWarpBad(const ArrayKd& g2, const ArrayKd& r2) const noexcept {
    return (r2 > (cfg.grad_factor * g2)).count() > cfg.max_outliers;
  }

  /// @brief Compute disparity
  void ApplyDisp(Eigen::Ref<Matrix2Kd> uv1s,
                 const ArrayKd& disps) const noexcept {
    uv1s.row(0).array() -= camera.Idepth2Disp(disps);
  }

  static void ExtractState(const FrameState& state0,
                           const FrameState& state1,
                           Eigen::Isometry3d& T_1_0,
                           Eigen::Array3d& eas,
                           Eigen::Array3d& bs);

  // data
  Camera camera;                // camera at this level/scale
  cv::Mat gray1l;               // gray0 of frame1
  cv::Mat gray1r;               // gray1 of frame1
  Eigen::Isometry3d T10;        // transform from 0 to 1
  Eigen::Array3d eas{0, 0, 0};  // exp^(a0l, a1l, a1r)
  Eigen::Array3d bs{0, 0, 0};   // (b0l, b1l, b1r)
  DirectCostCfg cfg;
};

/// @brief Base class for all direct methods (align, adjust, refine)
struct DirectMethod {
  DirectCfg cfg_;
  std::vector<cv::Range> pranges_{};

  DirectMethod() = default;
  explicit DirectMethod(const DirectCfg& cfg) : cfg_{cfg} { cfg_.Check(); }

  const DirectCfg& cfg() const noexcept { return cfg_; }

  static std::string LogIter(const std::pair<int, int>& level,
                             const std::pair<int, int>& iter,
                             const std::pair<int, double>& num_cost,
                             const std::pair<double, double>& xs,
                             double x2);
  static std::string LogIter(const std::pair<int, int>& level,
                             const std::pair<int, int>& iter,
                             const std::pair<int, double>& num_cost);

  static std::string LogConverge(int level, const DirectStatus& status);
};

/// @brief Transform points with the same scale using SE3
template <int N>
MatrixMNd<3, N> TransformScaled(const Eigen::Isometry3d& tf,
                                const MatrixMNd<3, N>& pt,
                                double s) noexcept {
  return (tf.linear() * pt).colwise() + tf.translation() * s;
}

/// @brief Warp function
/// @return uv1
template <int N>
MatrixMNd<2, N> Warp(const MatrixMNd<2, N>& uv0,
                     const Eigen::Array4d& fc0,
                     double q0,
                     const Eigen::Isometry3d& tf_1_0,
                     const Eigen::Array4d& fc1) noexcept {
  static_assert(N >= 1, "N must be >= 1");
  const auto nc0 = PnormFromPixel(uv0, fc0);
  const auto nh0 = Homogenize(nc0);
  const auto pt1 = TransformScaled(tf_1_0, nh0, q0);
  const auto nc1 = Project(pt1);
  const auto uv1 = PixelFromPnorm(nc1, fc1);
  return uv1;
}

inline double FastPoint5Pow(int n) noexcept {
  //  static constexpr std::array<double, 4> res = {1.0, 0.5, 0.25, 0.125};
  static constexpr std::array<double, 4> res = {1.0, 0.25, 0.0625, 0.015625};
  return res[n];
}

}  // namespace sv::dsol

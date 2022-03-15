#pragma once

#include "sv/dsol/frame.h"

namespace sv::dsol {

struct DirectSolveCfg {
  int max_iters{4};         // num max iters per level
  int max_levels{0};        // num levels used, 0 means all levels
  double rel_change{0.01};  // relative change of cost to stop early

  void Check() const;
  std::string Repr() const;

  /// @brief Get actual num levels to use
  int GetNumLevels(int levels) const noexcept {
    return max_levels > 0 ? std::min(max_levels, levels) : levels;
  }
};

struct DirectCostCfg {
  bool affine{false};       // use affine brightness model
  bool stereo{false};       // use stereo images
  int c2{4};                // gradient weight c^2, w_g = c^2 / (c^2 + dI^2)
  int dof{4};               // dof for t-dist, w_r = (dof + 1) / (dof + r^2)
  int max_outliers{1};      // max outliers in each patch
  double grad_factor{1.0};  // r^2 > grad_factor * g^2 is outlier
  double min_depth{0.5};    // min depth to skip when project

  void Check() const;
  std::string Repr() const;

  /// @brief Get actual frame dim
  int GetFrameDim() const noexcept {
    return Dim::kPose + Dim::kAffine * static_cast<int>(affine) *
                            (static_cast<int>(stereo) + 1);
  }
};

/// @brief Config of direct method
struct DirectCfg {
  DirectSolveCfg solve;
  DirectCostCfg cost;

  void Check() const;
  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const DirectCfg& rhs) {
    return os << rhs.Repr();
  }
};

/// @brief Status of direct method
struct DirectStatus {
  int num_kfs{};       // num kfs used
  int num_points{};    // num points used
  int num_levels{};    // num levels used
  int max_iters{};     // max iters possible
  int num_iters{};     // num iters run
  int num_costs{};     // num costs
  double last_cost{};  // last costs
  double last_dx2{};   // last delta x sqnorm

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const DirectStatus& rhs) {
    return os << rhs.Repr();
  }
};

/// @brief Cost for direct method
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
  double CalcWeight(double g2, double r2, double wi = 1.0) const noexcept {
    const auto wg = cfg.c2 / (cfg.c2 + g2);               // gradient weight
    const auto wr = (cfg.dof + 1) / (cfg.dof + r2 * wg);  // student t weight
    return wg * wr * wi;
  }

  ArrayKd CalcWeight(const ArrayKd& g2,
                     const ArrayKd& r2,
                     double wi = 1.0) const noexcept {
    const ArrayKd wg = cfg.c2 / (cfg.c2 + g2);
    return wg * (wi * (cfg.dof + 1)) / (cfg.dof + r2 * wg);
  }

  /// @brief Check if this warp is ok or not
  /// @details If residual bigger than gradient * grad_factor
  bool IsWarpBad(const ArrayKd& g2, const ArrayKd& r2) const noexcept {
    return ((r2 > cfg.grad_factor * g2).count() > cfg.max_outliers);
  }

  /// @brief Compute disparity
  void ApplyDisp(Eigen::Ref<Matrix2Kd> uv1s,
                 const ArrayKd& z1ps,
                 double q0) const noexcept {
    // Warp to right image is simply subtracting ux by disparity d
    // disparity is computed by d = f * b / z, but since our warping has no
    // scale, we need to scale it back.
    //
    // z1 = z1' / q0 -> q1 = q0 / z1'
    uv1s.row(0).array() -= camera.Idepth2Disp((q0 / z1ps).eval());
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

  /// @brief Update status and check for early stop
  bool OnIterEnd(DirectStatus& status,
                 int level,
                 int iter,
                 int num,
                 double cost,
                 double dx2) const noexcept;

  /// @brief Prepare points, only consider point with info >= min_info
  int PointsInfoGe(KeyframePtrSpan keyframes, double min_info);
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

}  // namespace sv::dsol

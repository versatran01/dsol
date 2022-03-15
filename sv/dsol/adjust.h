#pragma once

#include <mutex>

#include "sv/dsol/direct.h"
#include "sv/dsol/frame.h"
#include "sv/dsol/hessian.h"

namespace sv::dsol {

using AdjustCfg = DirectCfg;
using AdjustStatus = DirectStatus;

/// @brief Warp and build Hessian
struct AdjustCost final : public DirectCost {
  Eigen::Vector2i ij{-1, -1};
  FramePointHessian* pblock{nullptr};
  // fej
  Eigen::Isometry3d T10_fej;        // T10 first estimate
  Eigen::Array3d eas_fej{0, 0, 0};  // exp^(a0l, a1l, a1r)
  Eigen::Array3d bs_fej{0, 0, 0};   // (b0l, b1l, b1r)
  double dinfo{1.0};                // delta info for each warped point

  /// @brief A simple struct to store Jacobians
  struct JacGeo {
    FramePoint::Matrix26d du1_dx0;
    FramePoint::Matrix26d du1_dx1;
    Eigen::Vector2d du1_dq0;
  };

  AdjustCost() = default;
  AdjustCost(int level,
             const Camera& camera_scaled,
             const Keyframe& kf0,
             const Keyframe& kf1,
             const Eigen::Vector2i& iandj,
             FramePointHessian& block,
             const DirectCostCfg& cost_cfg,
             double delta_info);

  FrameHessian2 Build(const FramePointGrid& points0,
                      const PatchGrid& patches0,
                      int gsize = 0) const noexcept;

  /// @brief Warp patch given point into target frame and update Hessian
  /// @return inlier (1), oob (0), outlier (-1)
  int WarpPatch(const Patch& patch0,
                const FramePoint& point0,
                FrameHessian2& hess01) const noexcept;

  JacGeo CalcJacGeo(const FramePoint& point0,
                    const Eigen::Vector3d& pt1l,
                    int cam_ind) const noexcept;

  bool UpdateHess(const Patch& patch0,
                  const FramePoint& point0,
                  const Patch& patch1,
                  const Eigen::Vector3d& pt1l,
                  FrameHessian2& hess01,
                  int cam_ind) const noexcept;
};

/// @brief Photometric Bundle Adjustment
class BundleAdjuster final : public DirectMethod {
  FramePointHessian block_{};
  SchurFrameHessian schur_{};
  PriorFrameHessian prior_{};

 public:
  using DirectMethod::DirectMethod;

  /// @brief Repr
  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const BundleAdjuster& rhs) {
    return os << rhs.Repr();
  }

  const FramePointHessian& block() const noexcept { return block_; }
  const SchurFrameHessian& schur() const noexcept { return schur_; }
  const PriorFrameHessian& prior() const noexcept { return prior_; }

  /// @brief Main bundle adjustment method
  AdjustStatus Adjust(KeyframePtrSpan keyframes,
                      const Camera& camera,
                      int gsize = 0);

  AdjustStatus AdjustImpl(KeyframePtrSpan keyframes,
                          const Camera& camera,
                          const DirectSolveCfg& solve_cfg,
                          const DirectCostCfg& cost_cfg,
                          int gsize = 0);

  /// @brief Marginalize frame
  /// @return Number of points marginalized
  void Marginalize(KeyframePtrSpan keyframes,
                   const Camera& camera,
                   int k2rm,
                   int gsize = 0);

  /// @brief Resever space for Hessian and Prior
  size_t Allocate(int max_frames, int max_points_per_frame);

  /// @brief Reset prior
  void ResetPrior() { prior_.ResetData(); }

 private:
  /// @brief Build Hessian for a single level
  void BuildLevel(KeyframePtrSpan keyframes,
                  const Camera& camera,
                  int level,
                  int gsize = 0);

  /// @brief Build Hessian for a single level of one keyframe
  /// @param camera is already the scaled camera at level i
  void BuildLevelKf(KeyframePtrSpan keyframes,
                    const Camera& camera_scaled,
                    int level,
                    int k0,
                    std::mutex& mtx,
                    int gsize = 0);

  void Solve(bool fix_scale, int gsize = 0);
  void Update(KeyframePtrSpan keyframes, int gsize = 0);
};

}  // namespace sv::dsol

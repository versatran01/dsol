#pragma once

#include "sv/dsol/direct.h"
#include "sv/dsol/hessian.h"

namespace sv::dsol {

using AlignCfg = DirectCfg;
using AlignStatus = DirectStatus;

using AlignCostCfg = DirectCostCfg;

/// @brief Cost for align
struct AlignCost final : public DirectCost {
  using JacGeo = FramePoint::Matrix26d;

  AlignCost() = default;
  AlignCost(int level,
            const Camera& camera_scaled,
            const Keyframe& keyframe,
            const Frame& frame,
            const DirectCostCfg& cost_cfg);

  /// @brief Build FrameHessian
  FrameHessian1 Build(const FramePointGrid& points0,
                      const PatchGrid& patches0,
                      DepthPointGrid& points1,
                      int gsize = 0) const;

  /// @brief Warp patch into target frame
  /// @return DepthPoint in target frame
  DepthPoint WarpPatch(const Patch& patch0,
                       const FramePoint& point0,
                       FrameHessian1& hess) const noexcept;

  JacGeo CalcJacGeo(const FramePoint& point0, int cam_ind) const noexcept;

  /// @brief Update Hessian from a single patch
  /// @return Whether this patch is an inlier
  bool UpdateHess(const Patch& patch0,
                  const FramePoint& point0,
                  const Patch& patch1,
                  FrameHessian1& hess,
                  int cam_ind = 0) const noexcept;
};

/// @brief Direct image alignment wrt keyframes
class FrameAligner final : public DirectMethod {
  std::vector<DepthPointGrid> points1_vec_{};

 public:
  using DirectMethod::DirectMethod;

  /// @brief Repr
  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const FrameAligner& rhs) {
    return os << rhs.Repr();
  }

  const auto& points1_vec() const noexcept { return points1_vec_; }

  /// @brief Align frame to a set of keyframes, updates frame parameters
  /// @return Number of total costs in the level 0 (full res)
  AlignStatus Align(KeyframePtrSpan keyframes,
                    const Camera& camera,
                    Frame& frame,
                    int gsize = 0);

  AlignStatus AlignImpl(KeyframePtrSpan keyframes,
                        const Camera& camera,
                        Frame& frame,
                        const DirectSolveCfg& solve_cfg,
                        const DirectCostCfg& cost_cfg,
                        int gsize = 0);

  /// @brief Allocate points1
  size_t Allocate(size_t num_kfs, const cv::Size& grid_size);

  /// @brief
  cv::Mat CalcCellIdepth(int cell_size) const;

 private:
  /// @brief Build Hessian for a signle level
  FrameHessian1 BuildLevel(KeyframePtrSpan keyframes,
                           const Camera& camera,
                           const Frame& frame,
                           int level,
                           int gsize = 0);
};

}  // namespace sv::dsol

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
  /// @return Number of inliers in this patch
  bool UpdateHess(const Patch& patch0,
                  const FramePoint& point0,
                  const Patch& patch1,
                  FrameHessian1& hess,
                  int cam_ind = 0) const noexcept;
};

/// @brief Direct image alignment wrt keyframes
class FrameAligner final : public DirectMethod {
  Frame::Vector10d x_, y_;
  std::vector<int> num_tracks_{};
  std::vector<DepthPointGrid> points1_vec_{};
  cv::Mat idepth_{};  // each element is [sum, cnt]

 public:
  using DirectMethod::DirectMethod;

  /// @brief Repr
  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const FrameAligner& rhs) {
    return os << rhs.Repr();
  }

  const auto& idepths() const noexcept { return idepth_; }
  const auto& num_tracks() const noexcept { return num_tracks_; }
  const auto& points1_vec() const noexcept { return points1_vec_; }

  void Reset();

  /// @brief Align frame to a set of keyframes, updates frame parameters
  /// @return Number of total costs in the level 0 (full res)
  AlignStatus Align(KeyframePtrSpan keyframes,
                    const Camera& camera,
                    Frame& frame,
                    int gsize);

  /// @brief A robust version of Align
  /// @details This will first try to find a level that converges
  AlignStatus Align2(KeyframePtrSpan keyframes,
                     const Camera& camera,
                     Frame& frame,
                     int gsize);

  /// @brief Allocate points1
  size_t Allocate(size_t num_kfs, const cv::Size& grid_size);

 private:
  /// @brief Align frame for a single level
  AlignStatus AlignLevel(KeyframePtrSpan keyframes,
                         const Camera& camera,
                         Frame& frame,
                         int level,
                         int gsize = 0);

  /// @brief Build Hessian for a signle level
  FrameHessian1 BuildLevel(KeyframePtrSpan keyframes,
                           const Camera& camera,
                           const Frame& frame,
                           int level,
                           int gsize = 0);

  int PrepPoints(KeyframePtrSpan keyframes, int count);
  double Solve(const FrameHessian1& hess, int dim, double lambda = 0.0);
  void UpdateTrackAndIdepth(KeyframePtrSpan keyframes, const Frame& frame);
};

/// @brief
int CountIdepths(const cv::Mat& idepths);
void Proj2Idepth(const DepthPointGrid& points1,
                 const cv::Size& cell_size,
                 cv::Mat& idepths);
void Proj2Mask(const DepthPointGrid& points1,
               const cv::Size& cell_size,
               cv::Mat& mask);

}  // namespace sv::dsol

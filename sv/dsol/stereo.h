#pragma once

#include "sv/dsol/frame.h"

namespace sv::dsol {

struct StereoCfg {
  int half_rows{2};        // half rows of search window
  int half_cols{3};        // half cols of search window
  int match_level{3};      // match coarse level
  int refine_size{1};      // refine size
  double min_zncc{0.5};    // min zncc during refinement
  double min_depth{4.0};   // min depth -> max disp
  double best_ratio{0.8};  // filter score > best * best_ratio

  void Check() const;
  std::string Repr() const;

  cv::Point half_patch_size() const noexcept { return {half_cols, half_rows}; }
  cv::Size full_patch_size() const noexcept {
    return {2 * half_cols + 1, 2 * half_rows + 1};
  }

  cv::Range refine_range(int disp) const noexcept {
    return {disp - refine_size, disp + refine_size};
  }
};

/// @brief Disparity and score
struct DispZncc {
  int disp{-1};    // disp >= 0
  float zncc{-1};  // zncc in [-1, 1]
};

/// @brief A simple sparse stereo matcher that first finds candidate matches in
/// the lowest resolution image and gradually refines them using higher
/// resolution images.
class StereoMatcher {
  StereoCfg cfg_;
  cv::Mat disps_;  // int16_t, -1 is invalid

 public:
  explicit StereoMatcher(const StereoCfg& cfg = {}) : cfg_{cfg} {
    cfg_.Check();
  }

  std::string Repr() const;
  const auto& cfg() const noexcept { return cfg_; }
  const cv::Mat& disps() const noexcept { return disps_; }

  /// @brief Sparse stereo match
  /// @return number of matches
  int Match(const Keyframe& keyframe, const Camera& camera, int gsize = 0);
  int Match(const Frame& frame,
            const PixelGrid& pixels,
            const Camera& camera,
            int gsize = 0);

  /// @brief Allocate disparity storage
  /// @return number of bytes allocated
  size_t Allocate(const cv::Size& grid_size);

 private:
  /// @brief Exhaustive initial match at lowest resolution
  int MatchCoarse(const cv::Mat& gray0,
                  const cv::Mat& gray1,
                  const FramePointGrid& points0,
                  double scale,
                  int max_disp,
                  int gsize = 0);

  /// @brief Refine
  int MatchRefine(const cv::Mat& gray0,
                  const cv::Mat& gray1,
                  const FramePointGrid& points0,
                  double scale,
                  int gsize = 0);

  /// @brief Find best match for pxi in disp range
  DispZncc BestMatch(const cv::Point2i& pxi,
                     const cv::Range& range,
                     const cv::Mat& gray0,
                     const cv::Mat& gray1,
                     Eigen::ArrayXf& patch0,
                     Eigen::ArrayXf& patch1) noexcept;

  /// @brief Extract patch around px for Zncc
  void ExtractPatchZncc(const cv::Mat& gray,
                        const cv::Point2i& px,
                        Eigen::ArrayXf& patch) const noexcept;
};

/// @brief Extract a patch from mat at roi
void ExtractRoiArrayXf(const cv::Mat& mat,
                       const cv::Rect& roi,
                       Eigen::Ref<Eigen::ArrayXf> out);

template <typename D>
void ZeroMeanNormalize(Eigen::ArrayBase<D>& p,
                       float eps = std::numeric_limits<float>::epsilon()) {
  static_assert(std::is_floating_point_v<typename D::Scalar>,
                "Element must be floating point");
  p -= p.mean();
  const auto pnorm2 = p.matrix().squaredNorm();
  if (pnorm2 > eps * 0.01) {
    p /= std::sqrt(pnorm2);
  }
}

}  // namespace sv::dsol

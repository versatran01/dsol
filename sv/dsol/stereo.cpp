#include "sv/dsol/stereo.h"

#include "sv/dsol/pixel.h"
#include "sv/util/logging.h"
#include "sv/util/tbb.h"

namespace sv::dsol {

using ArrayXf = Eigen::ArrayXf;

/// ============================================================================
void StereoCfg::Check() const {
  CHECK_GT(half_rows, 0);
  CHECK_GT(half_cols, 0);
  CHECK_GT(min_depth, 0);
  CHECK_GT(refine_size, 0);
  CHECK_GT(best_ratio, 0);
  CHECK_LT(best_ratio, 1);
  CHECK_LT(min_zncc, 1);
}

std::string StereoCfg::Repr() const {
  return fmt::format(
      "StereoCfg(half_rows={}, half_cols={}, match_level={}, refine_size={}, "
      "min_depth={}, min_zncc={})",
      half_rows,
      half_cols,
      match_level,
      refine_size,
      min_depth,
      min_zncc);
}

/// ============================================================================
int StereoMatcher::Match(const Keyframe& keyframe,
                         const Camera& camera,
                         int gsize) {
  CHECK(keyframe.is_stereo());
  CHECK(camera.is_stereo());

  Allocate(keyframe.points().cvsize());
  disps_.setTo(-1);

  const int coarse_level = cfg_.match_level;
  CHECK_LT(coarse_level, keyframe.levels());
  const auto cam_s = camera.AtLevel(coarse_level);
  const auto max_disp =
      static_cast<int>(std::ceil(cam_s.Depth2Disp(cfg_.min_depth)));

  const auto n_matched = MatchCoarse(keyframe.grays_l().at(coarse_level),
                                     keyframe.grays_r().at(coarse_level),
                                     keyframe.points(),
                                     cam_s.scale(),
                                     max_disp,
                                     gsize);

  VLOG(1) << fmt::format("- [L {}] coarse_scale: {}, max_disp: {}, matches: {}",
                         coarse_level,
                         cam_s.scale(),
                         max_disp,
                         n_matched);

  int n_removed_sum = 0;
  for (int level = coarse_level - 1; level >= 0; --level) {
    const auto scale = PyrLevel2Scale(level);
    const auto n_removed = MatchRefine(keyframe.grays_l().at(level),
                                       keyframe.grays_r().at(level),
                                       keyframe.points(),
                                       scale,
                                       gsize);
    VLOG(2) << fmt::format("- [L {}] removed matches: {} ", level, n_removed);
    n_removed_sum += n_removed;
  }

  return n_matched - n_removed_sum;
}

int StereoMatcher::MatchCoarse(const cv::Mat& gray0,
                               const cv::Mat& gray1,
                               const FramePointGrid& points0,
                               double scale,
                               int max_disp,
                               int gsize) {
  CHECK_LT(scale, 1);
  CHECK_EQ(gray0.rows, gray1.rows);
  CHECK_EQ(gray0.cols, gray1.cols);

  const auto patch_size = cfg_.full_patch_size();
  const auto border = cfg_.half_cols;

  return ParallelReduce(
      {0, points0.rows(), gsize},
      0,
      [&](int gr, int& n_matched) {
        ArrayXf patch0(patch_size.area());
        ArrayXf patch1(patch_size.area());

        for (int gc = 0; gc < points0.cols(); ++gc) {
          // Skip if point is bad or already initialized
          const auto& point = points0.at(gr, gc);
          if (point.SkipInit()) continue;

          // Scale pixel to this level (need to convert to int)
          const auto pxi = RoundPix(ScalePix(point.px(), scale));

          // Get the max_disp to search by first computing the leftmost possible
          // pixel, then finding out the actual search disp
          const int left = std::max(border, pxi.x - max_disp);
          const int search_disp = pxi.x - left;

          // Skip if search range is too small
          if (search_disp <= 2) continue;

          const auto best =
              BestMatch(pxi, {0, search_disp}, gray0, gray1, patch0, patch1);

          // At this stage accept any match that has zncc >= 0
          if (best.zncc >= 0) {
            CHECK_GE(best.disp, 0);
            disps_.at<int16_t>(gr, gc) = static_cast<int16_t>(best.disp);
            ++n_matched;
            VLOG(3) << fmt::format(
                "grid: (r={}, c={}), px: (x={}, y={}), disp={}, ncc={}",
                gr,
                gc,
                pxi.x,
                pxi.y,
                best.disp,
                best.zncc);
          }
        }  // gc
      },   // gr
      std::plus<>{});
}

int StereoMatcher::MatchRefine(const cv::Mat& gray0,
                               const cv::Mat& gray1,
                               const FramePointGrid& points0,
                               double scale,
                               int gsize) {
  CHECK_LE(scale, 1);
  CHECK_EQ(gray0.rows, gray1.rows);
  CHECK_EQ(gray0.cols, gray1.cols);

  const auto patch_area = cfg_.full_patch_size().area();

  return ParallelReduce(
      {0, points0.rows(), gsize},
      0,
      [&](int gr, int& n_removed) {
        ArrayXf patch0(patch_area);
        ArrayXf patch1(patch_area);

        for (int gc = 0; gc < points0.cols(); ++gc) {
          // This is disp at one level lower resolution
          auto& prev_disp = disps_.at<int16_t>(gr, gc);
          if (prev_disp < 0) continue;

          const auto& point = points0.at(gr, gc);
          CHECK(point.PixelOk());

          // Scale pixel to this level
          const auto pxi = RoundPix(ScalePix(point.px(), scale));

          CHECK(!IsPixOut(gray0,
                          pxi + cv::Point(cfg_.refine_size, 0),
                          cfg_.half_patch_size()));

          const auto range = cfg_.refine_range(prev_disp * 2);
          const auto best = BestMatch(pxi, range, gray0, gray1, patch0, patch1);

          // Update disparity if result is good
          if (best.disp >= 0 && best.zncc >= cfg_.min_zncc) {
            prev_disp = static_cast<int16_t>(best.disp);
          } else {
            // otherwise invalidate it, so that we won't refine it further
            prev_disp = -1;
            ++n_removed;
          }
        }  // gc
      },   // gr
      std::plus<>{});
}

size_t StereoMatcher::Allocate(const cv::Size& grid_size) {
  if (disps_.empty()) {
    disps_.create(grid_size, CV_16SC1);
    disps_.setTo(-1);
  } else {
    CHECK_EQ(disps_.rows, grid_size.height);
    CHECK_EQ(disps_.cols, grid_size.width);
  }
  return disps_.total() * disps_.elemSize();
}

DispZncc StereoMatcher::BestMatch(const cv::Point2i& pxi,
                                  const cv::Range& range,
                                  const cv::Mat& gray0,
                                  const cv::Mat& gray1,
                                  ArrayXf& patch0,
                                  ArrayXf& patch1) noexcept {
  DispZncc best;
  ExtractPatchZncc(gray0, pxi, patch0);
  // Only look at a small neighbor around the current disp
  for (int disp = range.start; disp <= range.end; ++disp) {
    if (disp < 0) continue;

    ExtractPatchZncc(gray1, pxi - cv::Point(disp, 0), patch1);
    const auto zncc = (patch0 * patch1).sum();

    if (zncc > best.zncc) {
      best.disp = disp;
      best.zncc = zncc;
    }
  }
  return best;
}

void StereoMatcher::ExtractPatchZncc(const cv::Mat& gray,
                                     const cv::Point2i& pxi,
                                     ArrayXf& patch) const noexcept {
  const cv::Rect roi{pxi - cfg_.half_patch_size(), cfg_.full_patch_size()};
  ExtractRoiArrayXf(gray, roi, patch);
  ZeroMeanNormalize(patch);
}

std::string StereoMatcher::Repr() const {
  return fmt::format("StereoMatcher(cfg={})", cfg_.Repr());
}

/// ============================================================================
void ExtractRoiArrayXf(const cv::Mat& mat,
                       const cv::Rect& roi,
                       Eigen::Ref<ArrayXf> out) {
  CHECK_EQ(out.size(), roi.area());

  int k = 0;
  for (int r = 0; r < roi.height; ++r) {
    for (int c = 0; c < roi.width; ++c) {
      out[k++] = mat.at<uchar>(r + roi.y, c + roi.x);
    }
  }
}

}  // namespace sv::dsol

#pragma once

#include "sv/dsol/image.h"
#include "sv/dsol/point.h"
#include "sv/util/cmap.h"
#include "sv/util/math.h"  // Interval

namespace sv::dsol {

/// @brief Convert gray to rgb
void Gray2Bgr(const std::vector<cv::Mat>& grays, std::vector<cv::Mat>& bgrs);

/// @brief Display image pyramid
class PyramidDisplay {
  ImagePyramid images_;
  std::vector<cv::Point> offsets_;
  cv::Mat canvas_;

 public:
  PyramidDisplay() = default;
  explicit PyramidDisplay(const ImagePyramid& images) { SetImages(images); }
  void SetImages(const ImagePyramid& images);
  void Redraw();

  const cv::Mat& canvas() const noexcept { return canvas_; }
  int levels() const noexcept { return static_cast<int>(images_.size()); }

  cv::Mat LevelAt(int level) const;
  cv::Mat TopLevel() const { return LevelAt(0); }

  void Display(const std::string& name) const;

 private:
  /// @brief Generate offsets for accessing part of canvas (per pyramid level)
  void CalcOffsets();
  /// @brief Allocate canvas
  void AllocateCanvas();
};

/// @brief
void DrawRectangle(cv::Mat& mat,
                   const cv::Point& center,
                   const cv::Point& half_size,
                   const cv::Scalar& color,
                   int thickness = 1);

/// @brief Draw sparse depth, solid means inlier
void DrawDepthPoint(cv::Mat& mat,
                    const DepthPoint& point,
                    const ColorMap& cmap,
                    const IntervalD& idepth_range,
                    int dilate);
void DrawDepthPoints(cv::Mat mat,
                     const DepthPointGrid& points,
                     const ColorMap& cmap,
                     const IntervalD& idepth_range,
                     int dilate = 1);
void DrawFramePoints(cv::Mat mat,
                     const FramePointGrid& points,
                     const ColorMap& cmap,
                     const IntervalD& idepth_range,
                     int dilate = 1);

/// @brief Draw selected pixels as small squares
void DrawSelectedPixels(cv::Mat mat,
                        const PixelGrid& pixels,
                        const cv::Scalar& color,
                        int dilate = 1);

void DrawDisparities(cv::Mat mat,
                     const cv::Mat& disps,
                     const PixelGrid& pixels,
                     const ColorMap& cmap,
                     double max_disp,
                     int dilate = 1);

}  // namespace sv::dsol

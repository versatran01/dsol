#include "sv/dsol/viz.h"

#include "sv/dsol/pixel.h"
#include "sv/util/logging.h"
#include "sv/util/ocv.h"

namespace sv::dsol {

void PyramidDisplay::SetImages(const ImagePyramid& images) {
  // Check that input is not empty and is image pyramid
  CHECK(!images.empty());
  CHECK(IsImagePyramid(images));

  // convert to color
  if (images[0].type() != CV_8UC3) {
    Gray2Bgr(images, images_);
  } else {
    images_ = images;
  }

  AllocateCanvas();
  CalcOffsets();
  Redraw();
}

void PyramidDisplay::AllocateCanvas() {
  CHECK_GT(levels(), 0);
  const auto& image0 = images_.front();

  int canvas_rows = image0.rows;
  int canvas_cols = image0.cols;
  if (levels() > 1) canvas_rows += images_[1].rows;

  if (canvas_.rows != canvas_rows || canvas_.cols != canvas_cols) {
    canvas_ = cv::Mat::zeros(canvas_rows, canvas_cols, CV_8UC3);
  }
}

void PyramidDisplay::CalcOffsets() {
  CHECK_GT(levels(), 0);
  offsets_.resize(levels());
  offsets_[0] = {0, 0};

  const int row_offset = images_.front().rows;
  int col_offset = 0;
  for (int l = 1; l < levels(); ++l) {
    offsets_[l] = {col_offset, row_offset};
    col_offset += images_.at(l).cols;
  }
}

void PyramidDisplay::Redraw() {
  for (int l = 0; l < levels(); ++l) {
    const auto& image = images_.at(l);
    auto canvas_l = LevelAt(l);
    CHECK_EQ(canvas_l.rows, image.rows);
    CHECK_EQ(canvas_l.cols, image.cols);
    image.copyTo(canvas_l);
  }
}

void PyramidDisplay::Display(const std::string& name) const {
  Imshow(name, canvas_);
}

cv::Mat PyramidDisplay::LevelAt(int level) const {
  const auto& image = images_.at(level);
  const cv::Size size{image.cols, image.rows};
  const cv::Rect roi{offsets_.at(level), size};
  return cv::Mat(canvas_, roi);
}

void DrawDepthPoint(cv::Mat& mat,
                    const DepthPoint& point,
                    const ColorMap& cmap,
                    const IntervalD& idepth_range,
                    int dilate) {
  if (point.PixelBad() || point.DepthBad()) return;

  const auto pxi = RoundPix(point.px());
  if (IsPixOut(mat, pxi, dilate)) return;

  cv::Scalar bgr;
  Eigen::Map<Eigen::Vector3d> bgr_map(&bgr[0]);
  const double x = idepth_range.Normalize(point.idepth());
  bgr_map = cmap.GetBgr(x) * 255;

  // info ok is solid, otherwise hollow
  const int thickness = point.InfoOk() ? -1 : 1;
  DrawRectangle(mat, pxi, {dilate, dilate}, bgr, thickness);
}

void DrawDepthPoints(cv::Mat mat,
                     const DepthPointGrid& points,
                     const ColorMap& cmap,
                     const IntervalD& idepth_range,
                     int dilate) {
  CHECK_EQ(mat.type(), CV_8UC3);
  for (const auto& point : points) {
    DrawDepthPoint(mat, point, cmap, idepth_range, dilate);
  }
}

void DrawFramePoints(cv::Mat mat,
                     const FramePointGrid& points,
                     const ColorMap& cmap,
                     const IntervalD& idepth_range,
                     int dilate) {
  CHECK_EQ(mat.type(), CV_8UC3);
  for (const auto& point : points) {
    DrawDepthPoint(mat, point, cmap, idepth_range, dilate);
  }
}

void DrawSelectedPoints(cv::Mat mat,
                        const FramePointGrid& points,
                        const cv::Scalar& color,
                        int dilate) {
  CHECK_EQ(mat.type(), CV_8UC3);
  for (const auto& point : points) {
    if (point.PixelBad()) continue;
    DrawRectangle(mat, point.px(), {dilate, dilate}, color, -1);
  }
}

void DrawRectangle(cv::Mat& mat,
                   const cv::Point& center,
                   const cv::Point& half_size,
                   const cv::Scalar& color,
                   int thickness) {
  const auto pt1 = center - half_size;
  const auto pt2 = center + half_size;
  // it's a rectangle so just use LINE_4
  cv::rectangle(mat, pt1, pt2, color, thickness, cv::LINE_4);
}

void DrawSelectedPixels(cv::Mat mat,
                        const PixelGrid& pixels,
                        const cv::Scalar& color,
                        int dilate) {
  for (const auto& px : pixels) {
    if (IsPixBad(px)) continue;
    DrawRectangle(mat, px, {dilate, dilate}, color, -1);
  }
}

void DrawDisparities(cv::Mat mat,
                     const cv::Mat& disps,
                     const PixelGrid& pixels,
                     const ColorMap& cmap,
                     double max_disp,
                     int dilate) {
  CHECK_EQ(mat.type(), CV_8UC3);
  CHECK_EQ(disps.type(), CV_16SC1);
  CHECK_EQ(disps.rows, pixels.rows());
  CHECK_EQ(disps.cols, pixels.cols());

  for (int gr = 0; gr < pixels.rows(); ++gr) {
    for (int gc = 0; gc < pixels.cols(); ++gc) {
      const auto disp = disps.at<int16_t>(gr, gc);
      if (disp < 0) continue;
      const auto& px = pixels.at(gr, gc);

      cv::Scalar bgr;
      Eigen::Map<Eigen::Vector3d> bgr_map(&bgr[0]);
      // inlier is filled, outlier is hollow
      bgr_map = cmap.GetBgr(static_cast<double>(disp) / max_disp) * 255;
      DrawRectangle(mat, px, {dilate, dilate}, bgr, -1);
    }
  }
}

void Gray2Bgr(const std::vector<cv::Mat>& grays, std::vector<cv::Mat>& bgrs) {
  bgrs.resize(grays.size());
  for (size_t i = 0; i < grays.size(); ++i) {
    const auto& gray = grays.at(i);
    CHECK_EQ(gray.type(), CV_8UC1);
    cv::cvtColor(grays[i], bgrs[i], cv::COLOR_GRAY2BGR);
  }
}

void DrawProjBboxes(cv::Mat mat,
                    const std::array<cv::Point2d, 4>& polys,
                    const cv::Scalar& color,
                    int thickness) {
  for (int i = 0; i < polys.size(); ++i) {
    cv::line(
        mat, polys.at(i), polys.at((i + 1) % 4), color, thickness, cv::LINE_8);
  }
}

}  // namespace sv::dsol

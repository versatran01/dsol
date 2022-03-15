#include "sv/dsol/select.h"

#include "sv/dsol/pixel.h"
#include "sv/util/logging.h"
#include "sv/util/tbb.h"

namespace sv::dsol {

PixelGrad FindMaxGrad(const cv::Mat& image,
                      const cv::Rect& win,
                      const cv::Mat& mask,
                      int max_grad) noexcept {
  PixelGrad pxg{};

  for (int wr = 0; wr < win.height; ++wr) {
    for (int wc = 0; wc < win.width; ++wc) {
      const cv::Point2i px{wc + win.x, wr + win.y};

      // check if px in mask is occupied, if yes then skip
      if (!mask.empty() && mask.at<uchar>(px) > 0) continue;

      const auto grad = GradAtI<uchar>(image, px);
      const auto grad2 = PointSqNorm(grad);

      // Skip if grad is not bigger than saved
      if (grad2 < pxg.grad2) continue;

      // otherwise save max grad
      pxg.px = px;
      pxg.grad2 = grad2;

      // if grad is big enough in either direction then we can stop early
      if (std::abs(grad.x) >= max_grad || std::abs(grad.y) >= max_grad) {
        return pxg;
      }
    }
  }

  return pxg;
}

void CalcPixelGrads(const cv::Mat& image,
                    const cv::Mat& mask,
                    PixelGradGrid& pxgrads,
                    int max_grad,
                    int border,
                    int gsize) {
  if (!mask.empty()) {
    CHECK_EQ(mask.type(), CV_8UC1);
    CHECK_EQ(image.rows, mask.rows);
    CHECK_EQ(image.cols, mask.cols);
  }
  CHECK_GE(border, 0);
  CHECK(!pxgrads.empty());
  CHECK(!image.empty());
  CHECK_EQ(image.type(), CV_8UC1);

  const int cell_rows = image.rows / pxgrads.rows();
  const int cell_cols = image.cols / pxgrads.cols();

  // note that because here we skip grid border, if grid is default initialized,
  // some will have px (0, 0)
  ParallelFor({border, pxgrads.rows() - border, gsize},
              [&](int gr) {
                for (int gc = border; gc < pxgrads.cols() - border; ++gc) {
                  auto& pxg = pxgrads.at(gr, gc);

                  // window will exclude border to avoid oob checks
                  const cv::Rect win{gc * cell_cols + 1,
                                     gr * cell_rows + 1,
                                     cell_cols - 1,
                                     cell_rows - 1};
                  pxg = FindMaxGrad(image, win, mask, max_grad);
                }  // gc
              }    // gr
  );
}

/// ============================================================================
std::string SelectCfg::Repr() const {
  return fmt::format(
      "SelectCfg(sel_level={}, cell_size={}, min_grad={}, max_grad={}, "
      "nms_size={}, min_ratio={}, max_ratio={}, reselect={})",
      sel_level,
      cell_size,
      min_grad,
      max_grad,
      nms_size,
      min_ratio,
      max_ratio,
      reselect);
}

void SelectCfg::Check() const {
  CHECK_GE(sel_level, 0);
  CHECK_LT(sel_level, 2);
  CHECK_GT(min_grad, 0);
  CHECK_LT(min_grad, max_grad);
  CHECK_GE(nms_size, 0);
  CHECK_LE(nms_size, 2);
  CHECK_LT(min_ratio, max_ratio);
}

/// ============================================================================
int PixelSelector::Select(const ImagePyramid& grays, int gsize) {
  // Make sure pyramid has enough levels
  CHECK_GT(grays.size(), cfg_.sel_level);

  // Make sure cell size is large enough for top of pyramid
  const auto cell_too_small = cfg_.cell_size < std::pow(2, grays.size() - 1);
  VLOG_IF(1, cell_too_small)
      << fmt::format("Cell size {} is too small compared to pyramid levels {}",
                     cfg_.cell_size,
                     grays.size());

  // Allocate storage if needed
  Allocate(grays);
  std::fill(pixels_.begin(), pixels_.end(), cv::Point{-1, -1});
  std::fill(pxgrads_.begin(), pxgrads_.end(), PixelGrad{});

  // Select pixels at (maybe smaller) pyramid level first
  CalcPixelGrads(grays.at(cfg_.sel_level),
                 occ_mask_,
                 pxgrads_,
                 cfg_.max_grad,
                 grid_border_,
                 gsize);

  int n_pixels{};
  const auto gray_top = grays.at(0);
  const auto upscale = static_cast<int>(std::pow(2, cfg_.sel_level));

  // Do a first pass of selection using the current min_grad
  const auto n1 = SelectPixels(gray_top, upscale, cfg_.min_grad, gsize);
  n_pixels += n1;

  // Based on the number of pixels, determine how we should change min_grad
  const double ratio = static_cast<double>(n1) / pixels_.area();

  // Determine change direction
  int delta = 0;
  if (ratio < cfg_.min_ratio) {
    delta = -1;
  } else if (ratio > cfg_.max_ratio) {
    delta = 1;
  }

  // Do a 2nd round of selection for the rest of pixels with half of min_grad
  if (cfg_.reselect && delta < 0) {
    n_pixels += SelectPixels(gray_top, upscale, cfg_.min_grad / 2, gsize);
  }

  VLOG(1) << fmt::format(
      "- select: 1st={}, ratio={:.2f}% grad={}, 2nd={}, grad={}",
      n1,
      ratio * 100,
      cfg_.min_grad,
      n_pixels - n1,
      cfg_.min_grad / 2);

  // Update min_grad for next round
  cfg_.min_grad = std::clamp(cfg_.min_grad + delta * 2, 2, 32);
  VLOG_IF(1, delta != 0) << "change min grad to : " << cfg_.min_grad;
  return n_pixels;
}

int PixelSelector::SelectPixels(int min_grad) {
  int n_pixels{};
  const int min_grad2 = min_grad * min_grad;

  for (int i = 0; i < pixels_.area(); ++i) {
    // Skip already selected pixels
    auto& px = pixels_.at(i);
    if (!IsPixBad(px)) continue;

    // Skip pixels with too small gradient
    const auto& pxg = pxgrads_.at(i);
    if (pxg.grad2 < min_grad2) continue;

    px = pxg.px;
    ++n_pixels;
  }

  return n_pixels;
}

int PixelSelector::SelectPixels(const cv::Mat& gray,
                                int upscale,
                                int min_grad,
                                int gsize) {
  if (upscale == 1) return SelectPixels(min_grad);

  const int min_grad2 = min_grad * min_grad;
  return ParallelReduce(
      {0, pixels_.rows(), gsize},
      0,
      [&](int gr, int& n) {
        for (int gc = 0; gc < pixels_.cols(); ++gc) {
          auto& px = pixels_.at(gr, gc);
          if (!IsPixBad(px)) continue;

          const auto& pxg = pxgrads_.at(gr, gc);
          if (pxg.grad2 < min_grad2) continue;

          // Find max grad pixel within this small window
          const cv::Rect win{
              pxg.px.x * upscale, pxg.px.y * upscale, upscale, upscale};
          // The result should always be valid
          px = FindMaxGrad(gray, win).px;
          ++n;
        }  // gc
      },   // gr
      std::plus<>{});
}

int PixelSelector::CreateMask(absl::Span<const DepthPointGrid> points1s) {
  CHECK(!occ_mask_.empty());
  occ_mask_.setTo(0);
  const auto scale = std::pow(2, -cfg_.sel_level);
  int n_pixels = 0;
  for (const auto& points1 : points1s) {
    n_pixels += UpdateMask(points1, scale, cfg_.nms_size);
  }
  return n_pixels;
}

int PixelSelector::UpdateMask(const DepthPointGrid& points1,
                              double scale,
                              int dilate) {
  CHECK_GT(scale, 0);
  CHECK_LE(scale, 1);
  CHECK_GE(dilate, 0);

  int n_pixels = 0;  // number of masked out points
  for (const auto& point : points1) {
    if (!point.InfoOk()) continue;

    // scale to mask level, because grid is in full res
    const auto px_s = ScalePix(point.px(), scale);
    const auto px_i = RoundPix(px_s);
    // skip if oob
    if (IsPixOut(occ_mask_, px_i, dilate)) continue;

    // update mask
    n_pixels += static_cast<int>(
        MatSetWin<uchar>(occ_mask_, px_i, {dilate, dilate}, 255));
  }
  return n_pixels;
}

std::string PixelSelector::Repr() const {
  return fmt::format(
      "PixelSelector(cfg={}, grid_border={})", cfg_.Repr(), grid_border_);
}

size_t PixelSelector::Allocate(const cv::Size& top_size,
                               const cv::Size& sel_size) {
  // Allocate grid
  const cv::Size grid_size{top_size.width / cfg_.cell_size,
                           top_size.height / cfg_.cell_size};
  if (pixels_.empty()) {
    pixels_.resize(grid_size, {-1, -1});
    pxgrads_.resize(grid_size);
  } else {
    CHECK_EQ(pixels_.rows(), grid_size.height);
    CHECK_EQ(pixels_.cols(), grid_size.width);
  }

  // Allocate mask
  if (occ_mask_.empty()) {
    occ_mask_ = cv::Mat::zeros(sel_size, CV_8UC1);
  } else {
    CHECK_EQ(occ_mask_.rows, sel_size.height);
    CHECK_EQ(occ_mask_.cols, sel_size.width);
  }

  return occ_mask_.total() * occ_mask_.elemSize() +
         pixels_.size() * sizeof(cv::Point2d) +
         pxgrads_.size() * sizeof(PixelGrad);
}

size_t PixelSelector::Allocate(const ImagePyramid& grays) {
  const auto top = grays.at(0);
  const auto sel = grays.at(cfg_.sel_level);
  return Allocate({top.cols, top.rows}, {sel.cols, sel.rows});
}

}  // namespace sv::dsol

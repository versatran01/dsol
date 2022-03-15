#pragma once

#include <absl/types/span.h>

#include "sv/dsol/image.h"
#include "sv/dsol/point.h"

namespace sv::dsol {

struct PixelGrad {
  cv::Point2i px{-1, -1};
  double grad2{-1};  // gradient sq norm
};
using PixelGradGrid = Grid2d<PixelGrad>;

/// @brief find maximum gradient within a window
/// @param mask, > 0 means occupied, will skip
/// @details early stop if grad sq is greater than max_grad2
PixelGrad FindMaxGrad(const cv::Mat& image,
                      const cv::Rect& win,
                      const cv::Mat& mask = cv::Mat(),
                      int max_grad = 128) noexcept;

/// @brief Select pixels with large image gradient
void CalcPixelGrads(const cv::Mat& image,
                    const cv::Mat& mask,
                    PixelGradGrid& pxgrads,
                    int max_grad,
                    int border = 1,
                    int gsize = 0);

/// @brief Select pixels by searching within a small window
// void RefinePixels(const cv::Mat& image,
//                  PixelGrid& pixels,
//                  int upscale,
//                  int gsize = 0);

/// @brief Select config
struct SelectCfg {
  int sel_level{1};       // pyramid level for initial selection
  int cell_size{16};      // cell size in top level
  int min_grad{8};        // mininum grad to be selected
  int max_grad{64};       // wont keep searching if we found pix > max_grad
  int nms_size{1};        // nms size when creating mask
  double min_ratio{0.0};  // decrease min_grad when ratio < min_ratio
  double max_ratio{1.0};  // increase min_grad when ratio > max_ratio
  bool reselect{false};   // reselect if first round is two low

  std::string Repr() const;
  void Check() const;
};

/// @brief Pixel selector that finds pixels with large gradient
/// @return valid pixel must have both xy > 0
class PixelSelector {
  SelectCfg cfg_{};
  cv::Mat occ_mask_;       // occupancy mask, avoid selection where mask > 0
  PixelGrid pixels_;       // selected pixel in each grid
  PixelGradGrid pxgrads_;  // stores pixels and grad
  int grid_border_{1};     // grid border

 public:
  /// @brief Ctor
  explicit PixelSelector(const SelectCfg& cfg = {}) : cfg_{cfg} { cfg.Check(); }

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const PixelSelector& rhs) {
    return os << rhs.Repr();
  }

  const SelectCfg& cfg() const noexcept { return cfg_; }
  const cv::Mat& mask() const noexcept { return occ_mask_; }
  const PixelGrid& pixels() const noexcept { return pixels_; }
  cv::Size cvsize() const noexcept { return pixels_.cvsize(); }

  /// @brief Select pixels with large graidents
  /// @return Number of selected pixels
  /// @note Selection result is stored in grid()
  int Select(const ImagePyramid& grays, int gsize = 0);

  /// @brief Update projection mask from warped
  int CreateMask(absl::Span<const DepthPointGrid> points1s);

  /// @brief Allocate storage for mask and grid
  /// @return number of bytes allocated
  size_t Allocate(const cv::Size& top_size, const cv::Size& self_size);
  size_t Allocate(const ImagePyramid& grays);

 private:
  /// @brief Select pixels from pxgrads
  int SelectPixels(int min_grad);
  int SelectPixels(const cv::Mat& gray,
                   int upscale,
                   int min_grad,
                   int gsize = 0);

  int UpdateMask(const DepthPointGrid& points, double scale, int dilate);
};

}  // namespace sv::dsol

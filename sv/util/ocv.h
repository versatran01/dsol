#pragma once

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace sv {

/// @brief Get the corresponding cv enum value given type
/// @example cv_type<cv::Vec3f>::value == CV_32FC3
///          cv_type_v<cv::Vec3f> == CV_32FC3
template <typename T>
struct cv_type;

template <>
struct cv_type<uchar> {
  static constexpr int value = CV_8U;
};

template <>
struct cv_type<schar> {
  static constexpr int value = CV_8S;
};

template <>
struct cv_type<ushort> {
  static constexpr int value = CV_16U;
};

template <>
struct cv_type<short> {
  static constexpr int value = CV_16S;
};

template <>
struct cv_type<int> {
  static constexpr int value = CV_32S;
};

template <>
struct cv_type<float> {
  static constexpr int value = CV_32F;
};

template <>
struct cv_type<double> {
  static constexpr int value = CV_64F;
};

template <typename T, int N>
struct cv_type<cv::Vec<T, N>> {
  static constexpr int value = (CV_MAKETYPE(cv_type<T>::value, N));
};

template <typename T>
inline constexpr int cv_type_v = cv_type<T>::value;

/// @brief Convert cv::Mat::type() to string
/// @example CvTypeStr(CV_8UC1) == "8UC1"
std::string CvTypeStr(int type) noexcept;

/// @brief Repr for various cv types
std::string Repr(const cv::Mat& mat);
std::string Repr(const cv::Size& size);
std::string Repr(const cv::Range& range);

/// Range * d
inline cv::Range& operator*=(cv::Range& lhs, int d) noexcept {
  lhs.start *= d;
  lhs.end *= d;
  return lhs;
}
inline cv::Range operator*(cv::Range lhs, int d) noexcept { return lhs *= d; }

/// Range / d
inline cv::Range& operator/=(cv::Range& lhs, int d) noexcept {
  lhs.start /= d;
  lhs.end /= d;
  return lhs;
}
inline cv::Range operator/(cv::Range lhs, int d) noexcept { return lhs /= d; }

/// Size / Size
inline cv::Size& operator/=(cv::Size& lhs, const cv::Size& rhs) noexcept {
  lhs.width /= rhs.width;
  lhs.height /= rhs.height;
  return lhs;
}

inline cv::Size operator/(cv::Size lhs, const cv::Size& rhs) noexcept {
  return lhs /= rhs;
}

inline cv::Mat CvZeroLike(const cv::Mat& mat) {
  return cv::Mat::zeros(mat.rows, mat.cols, mat.type());
}

/// @brief Apply color map to mat
/// @details input must be 1-channel, assume after scale the max will be 1
/// default cmap is 10 = PINK. For float image it will set nan to bad_color
cv::Mat ApplyCmap(const cv::Mat& input,
                  double scale = 1.0,
                  int cmap = cv::COLORMAP_PINK,
                  uint8_t bad_color = 255);

static constexpr int kImshowFlag = cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO;

/// @brief Create a window with name and show mat
void Imshow(const std::string& name,
            const cv::Mat& mat,
            int flag = kImshowFlag,
            cv::Point offset = {0, 0});

/// @brief A simple keyboard control using opencv waitKey
/// @details [space] - pause/resume, [s] - step, [p] - pause
class KeyControl {
 public:
  explicit KeyControl(int wait_ms = 0, const cv::Size& size = {256, 32});

  /// @brief This will block if state is paused
  bool Wait();

  int counter() const noexcept { return counter_; }

 private:
  bool paused_{true};
  int counter_{0};
  int wait_ms_{0};
  std::string name_{"control"};

  cv::Mat display_;
  cv::Scalar color_pause_{CV_RGB(255, 0, 0)};  // red
  cv::Scalar color_step_{CV_RGB(0, 0, 255)};   // blue
  cv::Scalar color_run_{CV_RGB(0, 255, 0)};    // green
};

/// @brief Tile cv window
class WindowTiler {
 public:
  explicit WindowTiler(const cv::Size& screen_size = {1440, 900},
                       const cv::Point& offset = {400, 400},
                       const cv::Point& start = {0, 0});

  /// @brief Tile at the next location
  void Tile(const std::string& name,
            const cv::Mat& mat,
            int flag = kImshowFlag);

  /// @brief Reset curr to start
  void Reset() { curr_ = start_; }

 private:
  /// @brief Compuet next move location on screen
  void Next() noexcept;

  cv::Size screen_size_;  // display screen size
  cv::Point offset_;      // offset per image
  cv::Point start_;       // starting point
  cv::Point curr_;        // current point to put image
};

}  // namespace sv

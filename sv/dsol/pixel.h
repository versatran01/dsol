#pragma once

#include <Eigen/Core>
#include <opencv2/core/mat.hpp>

namespace sv::dsol {

/// @brief Scale pixel, assume center of top left corner is (0, 0)
inline cv::Point2d ScalePix(const cv::Point2d& px, double scale) noexcept {
  constexpr double kHalfPix = 0.5;
  return {scale * (px.x + kHalfPix) - kHalfPix,
          scale * (px.y + kHalfPix) - kHalfPix};
}

/// @brief Check if pixel is outside of size by border
inline bool IsPixOut(const cv::Size& size,
                     const cv::Point2d& px,
                     const cv::Point2d& border) noexcept {
  return (px.x < border.x) || (px.y < border.y) ||
         (px.x > (size.width - border.x - 1)) ||
         (px.y > (size.height - border.y - 1));
}

inline bool IsPixOut(const cv::Size& size,
                     const cv::Point2d& px,
                     double border = 0) noexcept {
  return IsPixOut(size, px, {border, border});
}

inline bool IsPixOut(const cv::Mat& mat,
                     const cv::Point2d& px,
                     double border = 0) noexcept {
  return IsPixOut({mat.cols, mat.rows}, px, {border, border});
}

inline bool IsPixOut(const cv::Mat& mat,
                     const cv::Point2d& px,
                     const cv::Point2d& border) noexcept {
  return IsPixOut({mat.cols, mat.rows}, px, border);
}

/// @brief Check if pixel is inside of size by border
inline bool IsPixIn(const cv::Size& size,
                    const cv::Point2d& px,
                    double border = 0) noexcept {
  return !IsPixOut(size, px, border);
}

inline bool IsPixIn(const cv::Mat& mat,
                    const cv::Point2d& px,
                    double border = 0) noexcept {
  return IsPixIn({mat.cols, mat.rows}, px, border);
}

/// @brief Intensity accessor at integer pixel location
template <typename T>
double ValAtI(const cv::Mat& mat, const cv::Point& px) noexcept {
  return mat.at<T>(px);
}

/// @brief Intensity accessor, Bilinear Interpolation
/// @details https://en.wikipedia.org/wiki/Bilinear_interpolation
template <typename T>
double ValAtD(const cv::Mat& mat, const cv::Point2d& px) noexcept {
  const int x0i = static_cast<int>(std::floor(px.x));
  const int x1i = static_cast<int>(std::ceil(px.x));
  const int y0i = static_cast<int>(std::floor(px.y));
  const int y1i = static_cast<int>(std::ceil(px.y));

  const double f00 = mat.at<T>(y0i, x0i);
  const double f10 = mat.at<T>(y0i, x1i);
  const double f01 = mat.at<T>(y1i, x0i);
  const double f11 = mat.at<T>(y1i, x1i);

  // normalize coordinate to [0, 1]
  const auto x0 = px.x - x0i;
  const auto y0 = px.y - y0i;
  const auto x1 = 1.0 - x0;
  const auto y1 = 1.0 - y0;

  return f00 * x1 * y1 + f10 * x0 * y1 + f01 * x1 * y0 + f11 * x0 * y0;
}

/// @brief Gradient accessor, using central diff
template <typename T>
double GradXAtI(const cv::Mat& mat, const cv::Point& px) noexcept {
  const double l = mat.at<T>(px.y, px.x - 1);
  const double r = mat.at<T>(px.y, px.x + 1);
  return (r - l) / 2.0;
}

template <typename T>
double GradYAtI(const cv::Mat& mat, const cv::Point& px) noexcept {
  const double u = mat.at<T>(px.y - 1, px.x);
  const double d = mat.at<T>(px.y + 1, px.x);
  return (d - u) / 2.0;
}

template <typename T>
cv::Point2d GradAtI(const cv::Mat& mat, const cv::Point& px) noexcept {
  return {GradXAtI<T>(mat, px), GradYAtI<T>(mat, px)};
}

template <typename T>
double GradXAtD(const cv::Mat& mat, const cv::Point2d& px) noexcept {
  const double l = ValAtD<T>(mat, {px.x - 1, px.y});
  const double r = ValAtD<T>(mat, {px.x + 1, px.y});
  return (r - l) / 2.0;
}

template <typename T>
double GradYAtD(const cv::Mat& mat, const cv::Point2d& px) noexcept {
  const double u = ValAtD<T>(mat, {px.x, px.y - 1});
  const double d = ValAtD<T>(mat, {px.x, px.y + 1});
  return (d - u) / 2.0;
}

template <typename T>
cv::Point2d GradAtD(const cv::Mat& mat, const cv::Point2d& px) noexcept {
  return {GradXAtD<T>(mat, px), GradYAtD<T>(mat, px)};
}

template <typename T>
double GradXAtD2(const cv::Mat& mat, const cv::Point2d& px) noexcept {
  const double l2 = ValAtD<T>(mat, {px.x - 2, px.y});
  const double l1 = ValAtD<T>(mat, {px.x - 1, px.y});
  const double r1 = ValAtD<T>(mat, {px.x + 1, px.y});
  const double r2 = ValAtD<T>(mat, {px.x + 2, px.y});
  return (r2 + 3 * r1 - 3 * l1 - l2) / 8.0;
}

template <typename T>
double GradYAtD2(const cv::Mat& mat, const cv::Point2d& px) noexcept {
  const double u2 = ValAtD<T>(mat, {px.x, px.y - 2});
  const double u1 = ValAtD<T>(mat, {px.x, px.y - 1});
  const double d1 = ValAtD<T>(mat, {px.x, px.y + 1});
  const double d2 = ValAtD<T>(mat, {px.x, px.y + 2});
  return (d2 + 3 * d1 - 3 * u1 - u2) / 8.0;
}

template <typename T>
cv::Point2d GradAtD2(const cv::Mat& mat, const cv::Point2d& px) noexcept {
  return {GradXAtD2<T>(mat, px), GradYAtD2<T>(mat, px)};
}

/// @brief Gradient accessor, Sobel
template <typename T>
double SobelXAtI(const cv::Mat& mat, const cv::Point& px) noexcept {
  const double l0 = mat.at<T>(px.y - 1, px.x - 1);
  const double r0 = mat.at<T>(px.y - 1, px.x + 1);
  const double l1 = mat.at<T>(px.y, px.x - 1);
  const double r1 = mat.at<T>(px.y, px.x + 1);
  const double l2 = mat.at<T>(px.y + 1, px.x - 1);
  const double r2 = mat.at<T>(px.y + 1, px.x + 1);
  return (r0 + 2 * r1 + r2 - l0 - 2 * l1 - l2) / 8.0;
}

template <typename T>
double SobelYAtI(const cv::Mat& mat, const cv::Point& px) noexcept {
  const double u0 = mat.at<T>(px.y - 1, px.x - 1);
  const double u1 = mat.at<T>(px.y - 1, px.x);
  const double u2 = mat.at<T>(px.y - 1, px.x + 1);
  const double d0 = mat.at<T>(px.y + 1, px.x - 1);
  const double d1 = mat.at<T>(px.y + 1, px.x);
  const double d2 = mat.at<T>(px.y + 1, px.x + 1);
  return (d0 + 2 * d1 + d2 - u0 - 2 * u1 - u2) / 8.0;
}

template <typename T>
cv::Point2d SobelAtI(const cv::Mat& mat, const cv::Point& px) noexcept {
  return {SobelXAtI<T>(mat, px), SobelYAtI<T>(mat, px)};
}

/// @brief Gradient accessor, Sobel
template <typename T>
double ScharrXAtI(const cv::Mat& mat, const cv::Point& px) noexcept {
  const double l0 = mat.at<T>(px.y - 1, px.x - 1);
  const double r0 = mat.at<T>(px.y - 1, px.x + 1);
  const double l1 = mat.at<T>(px.y, px.x - 1);
  const double r1 = mat.at<T>(px.y, px.x + 1);
  const double l2 = mat.at<T>(px.y + 1, px.x - 1);
  const double r2 = mat.at<T>(px.y + 1, px.x + 1);
  return (3 * r0 + 10 * r1 + 3 * r2 - 3 * l0 - 10 * l1 - 3 * l2) / 32.0;
}

template <typename T>
double ScharrYAtI(const cv::Mat& mat, const cv::Point& px) noexcept {
  const double u0 = mat.at<T>(px.y - 1, px.x - 1);
  const double u1 = mat.at<T>(px.y - 1, px.x);
  const double u2 = mat.at<T>(px.y - 1, px.x + 1);
  const double d0 = mat.at<T>(px.y + 1, px.x - 1);
  const double d1 = mat.at<T>(px.y + 1, px.x);
  const double d2 = mat.at<T>(px.y + 1, px.x + 1);
  return (3 * d0 + 10 * d1 + 3 * d2 - 3 * u0 - 10 * u1 - 3 * u2) / 32.0;
}

template <typename T>
cv::Point2d ScharrAtI(const cv::Mat& mat, const cv::Point& px) noexcept {
  return {ScharrXAtI<T>(mat, px), ScharrYAtI<T>(mat, px)};
}

/// @brief Intensity and Gradient accessor, xy stores image gradient, z stores
/// intensity value
template <typename T>
cv::Point3d GradValAtD(const cv::Mat& mat, const cv::Point2d& px) noexcept {
  cv::Point3d out;
  // Note that we don't use ceil here, because we need to compute gradient from
  // these 4 pixels. Doing ceil would cause 0 and 1 to be the same pixel which
  // results in zero gradient
  const int x0i = static_cast<int>(std::floor(px.x));
  const int x1i = x0i + 1;
  const int y0i = static_cast<int>(std::floor(px.y));
  const int y1i = y0i + 1;

  const double f00 = mat.at<T>(y0i, x0i);
  const double f10 = mat.at<T>(y0i, x1i);
  const double f01 = mat.at<T>(y1i, x0i);
  const double f11 = mat.at<T>(y1i, x1i);

  // normalize coordinate to [0, 1]
  const auto x0 = px.x - x0i;
  const auto y0 = px.y - y0i;
  const auto x1 = 1.0 - x0;
  const auto y1 = 1.0 - y0;

  out.x = (f10 - f00) * y1 + (f11 - f01) * y0;
  out.y = (f01 - f00) * x1 + (f11 - f10) * x0;
  out.z = f00 * x1 * y1 + f10 * x0 * y1 + f01 * x1 * y0 + f11 * x0 * y0;
  return out;
}

/// @brief Round pixel from double to int
inline cv::Point2i RoundPix(const cv::Point2d& px) noexcept {
  return {static_cast<int>(std::lround(px.x)),
          static_cast<int>(std::lround(px.y))};
}

inline bool IsPixBad(const cv::Point& px) noexcept {
  return px.x <= 0 || px.y <= 0;
}

template <typename T>
double PointSqNorm(const cv::Point_<T>& p) noexcept {
  return p.x * p.x + p.y * p.y;
}

template <typename T>
double PointSqNorm(const cv::Point3_<T>& p) noexcept {
  return p.x * p.x + p.y * p.y + p.z * p.z;
}

/// ============================================================================
template <typename T>
double ValAtE(const cv::Mat& mat, const Eigen::Vector2d& uv) noexcept {
  const Eigen::Vector2d uv0 = uv.array().floor();
  const Eigen::Vector2d uv1 = uv.array().ceil();
  const Eigen::Vector2i uv0i = uv0.cast<int>();
  const Eigen::Vector2i uv1i = uv1.cast<int>();

  const double f00 = mat.at<T>(uv0i.y(), uv0i.x());
  const double f10 = mat.at<T>(uv0i.y(), uv1i.x());
  const double f01 = mat.at<T>(uv1i.y(), uv0i.x());
  const double f11 = mat.at<T>(uv1i.y(), uv1i.x());

  // normalize coordinate to [0, 1]
  const Eigen::Vector2d xy0 = uv - uv0;
  const Eigen::Vector2d xy1 = 1 - xy0.array();

  return f00 * xy1.x() * xy1.y() + f10 * xy0.x() * xy1.y() +
         f01 * xy1.x() * xy0.y() + f11 * xy0.x() * xy0.y();
}

template <typename T>
double GradXAtE(const cv::Mat& mat, const Eigen::Vector2d& uv) noexcept {
  const double l = ValAtE<T>(mat, {uv.x() - 1, uv.y()});
  const double r = ValAtE<T>(mat, {uv.x() + 1, uv.y()});
  return (r - l) / 2.0;
}

template <typename T>
double GradYAtE(const cv::Mat& mat, const Eigen::Vector2d& uv) noexcept {
  const double u = ValAtE<T>(mat, {uv.x(), uv.y() - 1});
  const double d = ValAtE<T>(mat, {uv.x(), uv.y() + 1});
  return (d - u) / 2.0;
}

template <typename T>
cv::Point2d GradAtE(const cv::Mat& mat, const Eigen::Vector2d& uv) noexcept {
  return {GradXAtE<T>(mat, uv), GradYAtE<T>(mat, uv)};
}

/// @brief Intensity and Gradient accessor, xy stores image gradient, z stores
/// intensity value
template <typename T>
Eigen::Vector3d GradValAtE(const cv::Mat& mat,
                           const Eigen::Vector2d& uv) noexcept {
  Eigen::Vector3d out;
  // Note that we don't use ceil here, because we need to compute gradient from
  // these 4 pixels. Doing ceil would cause 0 and 1 to be the same pixel which
  // results in zero gradient
  const Eigen::Vector2d uv0 = uv.array().floor();
  const Eigen::Vector2d uv1 = uv.array() + 1;
  const Eigen::Vector2i uv0i = uv0.cast<int>();
  const Eigen::Vector2i uv1i = uv1.cast<int>();

  const double f00 = mat.at<T>(uv0i.y(), uv0i.x());
  const double f10 = mat.at<T>(uv0i.y(), uv1i.x());
  const double f01 = mat.at<T>(uv1i.y(), uv0i.x());
  const double f11 = mat.at<T>(uv1i.y(), uv1i.x());

  // normalize coordinate to [0, 1]
  const Eigen::Vector2d xy0 = uv - uv0;
  const Eigen::Vector2d xy1 = 1 - xy0.array();

  out.x() = (f10 - f00) * xy1.y() + (f11 - f01) * xy0.y();
  out.y() = (f01 - f00) * xy1.x() + (f11 - f10) * xy0.x();
  out.z() = f00 * xy1.x() * xy1.y() + f10 * xy0.x() * xy1.y() +
            f01 * xy1.x() * xy0.y() + f11 * xy0.x() * xy0.y();
  return out;
}

}  // namespace sv::dsol

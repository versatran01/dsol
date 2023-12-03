#include "sv/dsol/camera.h"

#include <fmt/ostream.h>

#include "sv/util/logging.h"
#include "sv/util/tbb.h"

namespace sv::dsol {

using Array4d = Eigen::Array4d;

template <typename T>
T Sq(const T& v) {
  return v * v;
}

Array4d ScaleFxycxy(const Array4d& fxycxy, double scale) noexcept {
  if (scale == 1.0) return fxycxy;
  Array4d out;
  out.head<2>() = scale * fxycxy.head<2>();
  out.tail<2>() = scale * (fxycxy.tail<2>() + 0.5) - 0.5;
  return out;
}

MatrixMNd<2, 3> DprojDpoint(const Eigen::Vector3d& pt) noexcept {
  MatrixMNd<2, 3> J;
  const double z_inv = 1.0 / pt.z();
  const double z_inv2 = z_inv * z_inv;
  // clang-format off
  J << z_inv,     0, -pt.x() * z_inv2,
       0,     z_inv, -pt.y() * z_inv2;
  // clang-format on
  return J;
}

/// ============================================================================
Camera::Camera(const cv::Size& size,
               const Array4d& fxycxy,
               double baseline,
               double scale)
    : size_{size}, fxycxy_{fxycxy}, baseline_{baseline}, scale_{scale} {
  CHECK_GE(baseline_, 0);
  CHECK_GT(scale_, 0);
}

Camera Camera::FromMat(const cv::Size& size, const cv::Mat& intrin) {
  CHECK_EQ(intrin.total(), 5);
  CHECK_EQ(intrin.type(), CV_64FC1);
  Eigen::Map<const Eigen::Array4d> fxycxy_map(intrin.ptr<double>());
  return Camera{size, fxycxy_map, intrin.at<double>(4)};
}

std::string Camera::Repr() const {
  return fmt::format("Camera(w={}, h={}, fxycxy=[{}], b={}, scale={})",
                     size_.width,
                     size_.height,
                     fmt::streamed(fxycxy_.transpose()),
                     baseline_,
                     scale_);
}

Camera Camera::Scaled(double scale) const noexcept {
  if (scale == 1.0) return *this;
  return {
      cv::Size(std::ceil(size_.width * scale), std::ceil(size_.height * scale)),
      ScaleFxycxy(fxycxy_, scale),
      baseline_,
      scale_ * scale};
}

/// ============================================================================
VignetteModel::VignetteModel(const cv::Size& size,
                             const Eigen::Array2d& cxy,
                             const Eigen::Array3d& vs)
    : cxy_{cxy}, vs_{vs} {
  max_radius_ = cxy_.matrix().norm();
  map_.create(size, CV_64FC1);
  UpdateMap();
}

void VignetteModel::SetParams(const Eigen::Array3d& vs, int gsize) {
  vs_ = vs;
  if (Noop()) {
    map_.setTo(1);
    return;
  }
  UpdateMap(gsize);
}

void VignetteModel::UpdateMap(int gsize) {
  const auto xs = Eigen::ArrayXd::LinSpaced(map_.cols, 0, map_.cols - 1).eval();
  ParallelFor({0, map_.rows, gsize}, [&](int r) {
    const auto r2s =
        ((xs - cxy_.x()).square() + Sq(r - cxy_.y())) / Sq(max_radius_);
    Eigen::Map<Eigen::ArrayXd> map_row(map_.ptr<double>(r), map_.cols);
    map_row = 1 + vs_[0] * r2s.sqrt() + vs_[1] * r2s + vs_[2] * r2s.square();
  });
}

void VignetteModel::Correct(cv::Mat& gray) const {
  if (Noop()) return;
  CHECK_EQ(gray.type(), CV_8UC1);
  CHECK_EQ(gray.rows, map_.rows);
  CHECK_EQ(gray.cols, map_.cols);
  // TODO (dsol): verify correctness
  CHECK(false);
}

std::string VignetteModel::Repr() const {
  return fmt::format("VignetteModel(w={}, h={}, r={}, cxy=[{}], vs=[{}])",
                     map_.cols,
                     map_.rows,
                     max_radius_,
                     fmt::streamed(cxy_.transpose()),
                     fmt::streamed(vs_.transpose()));
}

}  // namespace sv::dsol

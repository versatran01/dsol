#pragma once

#include <glog/logging.h>

#include <Eigen/Cholesky>
#include <type_traits>

#include "sv/util/eigen.h"

namespace sv {

static constexpr auto kNaNF = std::numeric_limits<float>::quiet_NaN();
static constexpr auto kPiF = static_cast<float>(M_PI);
static constexpr auto kTauF = static_cast<float>(M_PI * 2);
static constexpr auto kPiD = static_cast<double>(M_PI);
static constexpr auto kTauD = static_cast<double>(M_PI * 2);

template <typename T>
constexpr T Sq(T x) noexcept {
  return x * x;
}

template <typename T>
T Deg2Rad(T deg) noexcept {
  static_assert(std::is_floating_point_v<T>, "T must be floating point");
  return deg / 180.0 * M_PI;
}

template <typename T>
T Rad2Deg(T rad) noexcept {
  static_assert(std::is_floating_point_v<T>, "T must be floating point");
  return rad / M_PI * 180.0;
}

/// @brief Precomputed sin and cos
template <typename T>
struct SinCos {
  static_assert(std::is_floating_point_v<T>, "T must be floating point");

  SinCos(T rad = 0) : sin{std::sin(rad)}, cos{std::cos(rad)} {}

  T sin{};
  T cos{};
};

using SinCosF = SinCos<float>;
using SinCosD = SinCos<double>;

/// @brief Polynomial approximation to asin
template <typename T>
T AsinApprox(T x) noexcept {
  static_assert(std::is_floating_point_v<T>, "T must be floating point");

  const T x2 = x * x;
  return x * (1 + x2 * (1 / 6.0 + x2 * (3.0 / 40.0 + x2 * 5.0 / 112.0)));
}

/// @brief A faster atan2
template <typename T>
T Atan2Approx(T y, T x) noexcept {
  static_assert(std::is_floating_point_v<T>, "T must be floating point");
  // http://pubs.opengroup.org/onlinepubs/009695399/functions/atan2.html
  // Volkan SALMA
  static constexpr T kPi3_4 = M_PI_4 * 3;
  static constexpr T kPi_4 = M_PI_4;

  T r, angle;
  T abs_y = fabs(y) + 1e-10;  // kludge to prevent 0/0 condition
  if (x < 0.0) {
    r = (x + abs_y) / (abs_y - x);
    angle = kPi3_4;
  } else {
    r = (x - abs_y) / (x + abs_y);
    angle = kPi_4;
  }
  angle += (0.1963 * r * r - 0.9817) * r;
  return y < 0.0 ? -angle : angle;
}

/// @struct Running mean and variance
template <typename T, int N>
struct MeanVar {
  using Vector = Eigen::Matrix<T, N, 1>;

  int n{0};
  Vector mean{Vector::Zero()};
  Vector var_sum_{Vector::Zero()};

  /// @brief compute covariance
  Vector Var() const noexcept { return var_sum_ / (n - 1); }

  /// @brief whether result is ok
  bool ok() const noexcept { return n > 1; }

  void Add(const Vector& x) noexcept {
    ++n;
    const Vector dx = x - mean;
    const Vector dx_n = dx / n;
    mean += dx_n;
    var_sum_.noalias() += (n - 1.0) * dx_n.cwiseProduct(dx);
  }

  void Reset() noexcept {
    n = 0;
    mean.setZero();
    var_sum_.setZero();
  }
};

using MeanVar3f = MeanVar<float, 3>;
using MeanVar3d = MeanVar<double, 3>;

/// @struct Running mean and covariance
template <typename T, int N>
struct MeanCovar {
  using Matrix = Eigen::Matrix<T, N, N>;
  using Vector = Eigen::Matrix<T, N, 1>;

  int n{0};
  Vector mean{Vector::Zero()};
  Matrix covar_sum_{Matrix::Zero()};

  /// @brief compute covariance
  Matrix Covar() const noexcept { return covar_sum_ / (n - 1); }

  /// @brief whether result is ok
  bool ok() const noexcept { return n > 1; }

  void Add(const Vector& x) noexcept {
    ++n;
    const Vector dx = x - mean;
    const Vector dx_n = dx / n;
    mean += dx_n;
    covar_sum_.noalias() += ((n - 1.0) * dx_n) * dx.transpose();
  }

  void Reset() noexcept {
    n = 0;
    mean.setZero();
    covar_sum_.setZero();
  }
};

using MeanCovar3f = MeanCovar<float, 3>;
using MeanCovar3d = MeanCovar<double, 3>;

/// @brief force the axis to be right handed for 3D
/// @details sometimes eigvecs has det -1 (reflection), this makes it a rotation
/// @ref
/// https://docs.ros.org/en/noetic/api/rviz/html/c++/covariance__visual_8cpp_source.html
void MakeRightHanded(Eigen::Vector3d& eigvals, Eigen::Matrix3d& eigvecs);

/// @brief Computes matrix square root using Cholesky A = LL' = U'U
template <typename T, int N>
Eigen::Matrix<T, N, N> MatrixSqrtUtU(const Eigen::Matrix<T, N, N>& A) {
  return A.template selfadjointView<Eigen::Upper>().llt().matrixU();
}

/// A (real) closed interval, boost interval is too heavy
template <typename T>
struct Interval {
  Interval() = default;
  Interval(const T& left, const T& right) noexcept
      : left_(left), right_(right) {
    CHECK_LE(left, right);
  }

  T left_, right_;

  const T& a() const noexcept { return left_; }
  const T& b() const noexcept { return right_; }
  T width() const noexcept { return b() - a(); }
  bool empty() const noexcept { return b() <= a(); }
  bool ContainsClosed(const T& v) const noexcept {
    return (a() <= v) && (v <= b());
  }
  bool ContainsOpen(const T& v) const noexcept {
    return (a() < v) && (v < b());
  }

  /// Whether this interval contains other
  bool ContainsClosed(const Interval<T>& other) const noexcept {
    return a() <= other.a() && other.b() <= b();
  }

  /// Normalize v to [0, 1], assumes v in [left, right]
  /// Only enable if we have floating type
  T Normalize(const T& v) const noexcept {
    static_assert(std::is_floating_point_v<T>, "Must be floating point");
    return (v - a()) / width();
  }

  /// InvNormalize v to [left, right], assumes v in [0, 1]
  T InvNormalize(const T& v) const noexcept {
    static_assert(std::is_floating_point_v<T>, "Must be floating point");
    return v * width() + a();
  }
};

using IntervalF = Interval<float>;
using IntervalD = Interval<double>;

}  // namespace sv

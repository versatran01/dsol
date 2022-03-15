#pragma once

#include <Eigen/Core>

namespace sv {

using MatrixXd = Eigen::MatrixXd;
using VectorXd = Eigen::VectorXd;

// Eigen::Ref typedefs
using MatrixXdRef = Eigen::Ref<Eigen::MatrixXd>;
using VectorXdRef = Eigen::Ref<Eigen::VectorXd>;
using MatrixXdCRef = Eigen::Ref<const Eigen::MatrixXd>;
using VectorXdCRef = Eigen::Ref<const Eigen::VectorXd>;

// Eigen::Map typedefs
using MatrixXdMap = Eigen::Map<Eigen::MatrixXd>;
using VectorXdMap = Eigen::Map<Eigen::VectorXd>;
using VectorXdCMap = Eigen::Map<const Eigen::VectorXd>;
using MatrixXdCMap = Eigen::Map<const Eigen::MatrixXd>;

template <int M, int N>
using MatrixMNd = Eigen::Matrix<double, M, N>;
template <int N>
using VectorNd = Eigen::Matrix<double, N, 1>;
template <int M, int N>
using ArrayMNd = Eigen::Array<double, M, N>;

/// @brief Rotate block i to the front of the system, assuming block size n
/// @details This is a stable rotation, meaning it won't change the order of the
/// rest of the blocks. For example, assuming the diagonal of a square matrix is
/// [0, 1, 2, 3, 4] and we want to rotate block 2 to top left. This will result
/// a new matrix with diagonal [2, 0, 1, 3, 4], instead of [2, 3, 4, 0, 1].
void StableRotateBlockTopLeft(MatrixXdRef H, VectorXdRef b, int i, int n);

/// @brief Fill upper/lower triangular part of M such that M is symmetric
void FillUpperTriangular(MatrixXdRef M);
void FillLowerTriangular(MatrixXdRef M);
void MakeSymmetric(MatrixXdRef M);

/// @brief Element wise inverse and set result of inf to c
inline void SafeCwiseInverse(VectorXdRef x, double c = 0) noexcept {
  x = x.cwiseInverse();
  x = x.array().isInf().select(c, x);
}

/// @brief Make skew symmetric matrix
inline Eigen::Matrix3d Hat3d(double x, double y, double z) noexcept {
  Eigen::Matrix3d S;
  // clang-format off
  S <<  0, -z,  y,
        z,  0, -x,
       -y,  x,  0;
  // clang-format on
  return S;
}

inline Eigen::Matrix3d Hat3d(const Eigen::Vector3d& w) noexcept {
  return Hat3d(w.x(), w.y(), w.z());
}

/// @brief Make skew symmetric matrix
template <typename T>
Eigen::Matrix<T, 3, 3> Hat3(const T& x, const T& y, const T& z) noexcept {
  static_assert(std::is_floating_point_v<T>, "T must be floating point");
  Eigen::Matrix<T, 3, 3> S;
  // clang-format off
  S << T(0.0),  -z,     y,
       z,       T(0.0), -x,
       -y,      x,      T(0.0);
  // clang-format on
  return S;
}

template <typename T>
Eigen::Matrix<T, 3, 3> Hat3(const Eigen::Matrix<T, 3, 1>& w) noexcept {
  return Hat3(w.x(), w.y(), w.z());
}

/// @brief Marginalize the top-left block onto the rest
/// @details Hsc must be symmetric, and as a result Hpr is also symmetric
void MargTopLeftBlock(const MatrixXdCRef& Hf,
                      const VectorXdCRef& bf,
                      MatrixXdRef Hm,
                      VectorXdRef bm,
                      int dim);

}  // namespace sv

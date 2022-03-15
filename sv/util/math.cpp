#include "sv/util/math.h"

#include <Eigen/Geometry>

namespace sv {

void MakeRightHanded(Eigen::Vector3d& eigvals, Eigen::Matrix3d& eigvecs) {
  auto hand = eigvecs.col(0).cross(eigvecs.col(1)).dot(eigvecs.col(2));
  if (hand < 0) {
    eigvecs.col(0).swap(eigvecs.col(1));
    eigvals.row(0).swap(eigvals.row(1));
  }
}

}  // namespace sv

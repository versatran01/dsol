#include "sv/ros1/msg_conv.h"

#include <fmt/ranges.h>

#include "sv/util/logging.h"

namespace sv {

using SO3d = Sophus::SO3d;
using SE3d = Sophus::SE3d;

namespace gm = geometry_msgs;
using sensor_msgs::PointField;

void Eigen2Ros(const Eigen::Vector3d& e, gm::Point& r) {
  r.x = e.x();
  r.y = e.y();
  r.z = e.z();
}

void Ros2Eigen(const gm::Point& r, Eigen::Ref<Eigen::Vector3d> e) {
  e.x() = r.x;
  e.y() = r.y;
  e.z() = r.z;
}

void Eigen2Ros(const Eigen::Vector3d& e, gm::Vector3& r) {
  r.x = e.x();
  r.y = e.y();
  r.z = e.z();
}

void Ros2Eigen(const gm::Vector3& r, Eigen::Ref<Eigen::Vector3d> e) {
  e.x() = r.x;
  e.y() = r.y;
  e.z() = r.z;
}

void Eigen2Ros(const Eigen::Quaterniond& e, gm::Quaternion& r) {
  r.w = e.w();
  r.x = e.x();
  r.y = e.y();
  r.z = e.z();
}

void Ros2Eigen(const gm::Quaternion& r, Eigen::Quaterniond& e) {
  e.w() = r.w;
  e.x() = r.x;
  e.y() = r.y;
  e.z() = r.z;
}

void Eigen2Ros(const Eigen::Vector3d& pos,
               const Eigen::Quaterniond& quat,
               gm::Pose& pose) {
  Eigen2Ros(pos, pose.position);
  Eigen2Ros(quat, pose.orientation);
}

void Eigen2Ros(const Eigen::Vector3d& pos,
               const Eigen::Quaterniond& quat,
               gm::Transform& tf) {
  Eigen2Ros(pos, tf.translation);
  Eigen2Ros(quat, tf.rotation);
}

void Ros2Eigen(const gm::Transform& tf, Eigen::Isometry3d& iso) {
  Ros2Eigen(tf.translation, iso.translation());
  Eigen::Quaterniond quat;
  Ros2Eigen(tf.rotation, quat);
  iso.linear() = quat.toRotationMatrix();
}

void Eigen2Ros(const Eigen::Isometry3d& iso, gm::Pose& pose) {
  Eigen2Ros(iso.translation(), pose.position);
  Eigen2Ros(Eigen::Quaterniond(iso.rotation()), pose.orientation);
}

void Ros2Ros(const gm::Pose& pose, gm::Transform& tf) {
  tf.translation.x = pose.position.x;
  tf.translation.y = pose.position.y;
  tf.translation.z = pose.position.z;
  tf.rotation = pose.orientation;
}

void Sophus2Ros(const SE3d& se3, gm::Pose& pose) {
  Eigen2Ros(se3.translation(), pose.position);
  Eigen2Ros(se3.unit_quaternion(), pose.orientation);
}

void Sophus2Ros(const Sophus::SE3d& se3, gm::Transform& tf) {
  Eigen2Ros(se3.translation(), tf.translation);
  Eigen2Ros(se3.unit_quaternion(), tf.rotation);
}

void Sophus2Ros(const Sophus::SO3d& so3, gm::Quaternion& quat) {
  Eigen2Ros(so3.unit_quaternion(), quat);
}

void Ros2Sophus(const gm::Quaternion& quat, Sophus::SO3d& so3) {
  Eigen::Quaterniond q;
  Ros2Eigen(quat, q);
  so3.setQuaternion(q);
}

void Ros2Sophus(const gm::Pose& pose, Sophus::SE3d& se3) {
  Ros2Sophus(pose.orientation, se3.so3());
  Ros2Eigen(pose.position, se3.translation());
}

/// @brief Get size of datatype from PointFiled
int GetPointFieldDataTypeBytes(uint8_t dtype) noexcept {
  // INT8 = 1u,
  // UINT8 = 2u,
  // INT16 = 3u,
  // UINT16 = 4u,
  // INT32 = 5u,
  // UINT32 = 6u,
  // FLOAT32 = 7u,
  // FLOAT64 = 8u,
  switch (dtype) {
    case 0U:
      return 0;
    case 1U:  // int8
    case 2U:  // uint8
      return 1;
    case 3U:  // int16
    case 4U:  // uint16
      return 2;
    case 5U:  // int32
    case 6U:  // uint32
    case 7U:  // float32
      return 4;
    case 8U:  // float64
      return 8;
    default:
      return 0;
  }
}

/// @brief Extract point step from
int GetPointStep(const PointFields& fields) {
  int step = 0;

  for (const auto& f : fields) {
    step += f.count * GetPointFieldDataTypeBytes(f.datatype);
  }

  return step;
}

Cloud2Helper::Cloud2Helper(int rows, int cols, const PointFields& fields) {
  cloud.fields = fields;
  cloud.point_step = GetPointStep(fields);
  resize(rows, cols);
}

void Cloud2Helper::resize(int rows, int cols) {
  CHECK(!cloud.fields.empty());
  CHECK_GT(cloud.point_step, 0);

  cloud.width = cols;
  cloud.height = rows;
  cloud.row_step = cloud.point_step * cloud.width;
  cloud.data.resize(rows * cols * cloud.point_step);
}

PointFields MakePointFields(const std::string& fstr) {
  std::vector<PointField> fields;
  fields.reserve(fstr.size());

  int offset{0};
  PointField field;

  for (auto s : fstr) {
    s = std::tolower(s);
    if (s == 'x' || s == 'y' || s == 'z') {
      field.name = s;
      field.offset = offset;
      field.datatype = PointField::FLOAT32;
      field.count = 1;
    } else if (s == 'i') {
      field.name = "intensity";
      field.offset = offset;
      field.datatype = PointField::FLOAT32;
      field.count = 1;
    } else {
      continue;
    }

    // update offset
    offset += GetPointFieldDataTypeBytes(field.datatype) * field.count;
    fields.push_back(field);
  }

  return fields;
}

}  // namespace sv

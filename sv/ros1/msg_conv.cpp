#include "sv/ros1/msg_conv.h"

namespace sv {

using sensor_msgs::PointField;

void Eigen2Ros(const Eigen::Vector3d& e, geometry_msgs::Point& r) {
  r.x = e.x();
  r.y = e.y();
  r.z = e.z();
}

void Ros2Eigen(const geometry_msgs::Point& r, Eigen::Ref<Eigen::Vector3d> e) {
  e.x() = r.x;
  e.y() = r.y;
  e.z() = r.z;
}

void Eigen2Ros(const Eigen::Vector3d& e, geometry_msgs::Vector3& r) {
  r.x = e.x();
  r.y = e.y();
  r.z = e.z();
}

void Ros2Eigen(const geometry_msgs::Vector3& r, Eigen::Ref<Eigen::Vector3d> e) {
  e.x() = r.x;
  e.y() = r.y;
  e.z() = r.z;
}

void Eigen2Ros(const Eigen::Quaterniond& e, geometry_msgs::Quaternion& r) {
  r.w = e.w();
  r.x = e.x();
  r.y = e.y();
  r.z = e.z();
}

void Ros2Eigen(const geometry_msgs::Quaternion& r, Eigen::Quaterniond& e) {
  e.w() = r.w;
  e.x() = r.x;
  e.y() = r.y;
  e.z() = r.z;
}

void Eigen2Ros(const Eigen::Vector3d& pos,
               const Eigen::Quaterniond& quat,
               geometry_msgs::Pose& pose) {
  Eigen2Ros(pos, pose.position);
  Eigen2Ros(quat, pose.orientation);
}

void Eigen2Ros(const Eigen::Vector3d& pos,
               const Eigen::Quaterniond& quat,
               geometry_msgs::Transform& tf) {
  Eigen2Ros(pos, tf.translation);
  Eigen2Ros(quat, tf.rotation);
}

void Ros2Eigen(const geometry_msgs::Transform& tf, Eigen::Isometry3d& iso) {
  Ros2Eigen(tf.translation, iso.translation());
  Eigen::Quaterniond quat;
  Ros2Eigen(tf.rotation, quat);
  iso.linear() = quat.toRotationMatrix();
}

void Eigen2Ros(const Eigen::Isometry3d& iso, geometry_msgs::Pose& pose) {
  Eigen2Ros(iso.translation(), pose.position);
  Eigen2Ros(Eigen::Quaterniond(iso.rotation()), pose.orientation);
}

void Ros2Ros(const geometry_msgs::Pose& pose, geometry_msgs::Transform& tf) {
  tf.translation.x = pose.position.x;
  tf.translation.y = pose.position.y;
  tf.translation.z = pose.position.z;
  tf.rotation = pose.orientation;
}

void Sophus2Ros(const Sophus::SE3d& se3, geometry_msgs::Pose& pose) {
  Eigen2Ros(se3.translation(), pose.position);
  Eigen2Ros(se3.unit_quaternion(), pose.orientation);
}

void Sophus2Ros(const Sophus::SE3d& se3, geometry_msgs::Transform& tf) {
  Eigen2Ros(se3.translation(), tf.translation);
  Eigen2Ros(se3.unit_quaternion(), tf.rotation);
}

void Sophus2Ros(const Sophus::SO3d& so3, geometry_msgs::Quaternion& quat) {
  Eigen2Ros(so3.unit_quaternion(), quat);
}

void Ros2Sophus(const geometry_msgs::Quaternion& quat, Sophus::SO3d& so3) {
  Eigen::Quaterniond q;
  Ros2Eigen(quat, q);
  so3.setQuaternion(q);
}

void Ros2Sophus(const geometry_msgs::Pose& pose, Sophus::SE3d& se3) {
  Ros2Sophus(pose.orientation, se3.so3());
  Ros2Eigen(pose.position, se3.translation());
}

std::vector<PointField> MakePointFieldsXYZI() noexcept {
  std::vector<PointField> fields(4);

  PointField field;
  field.name = "x";
  field.offset = 0;
  field.datatype = PointField::FLOAT32;
  field.count = 1;
  fields.push_back(field);

  field.name = "y";
  field.offset = 4;
  field.datatype = PointField::FLOAT32;
  field.count = 1;
  fields.push_back(field);

  field.name = "z";
  field.offset = 8;
  field.datatype = PointField::FLOAT32;
  field.count = 1;
  fields.push_back(field);

  field.name = "intensity";
  field.offset = 12;
  field.datatype = PointField::FLOAT32;
  field.count = 1;
  fields.push_back(field);

  return fields;
}

}  // namespace sv

#pragma once

#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Transform.h>
#include <sensor_msgs/PointCloud2.h>

#include <opencv2/core/core.hpp>
#include <sophus/se3.hpp>

namespace sv {

// Vector3d <-> Point
void Eigen2Ros(const Eigen::Vector3d& e, geometry_msgs::Point& r);
void Ros2Eigen(const geometry_msgs::Point& r, Eigen::Ref<Eigen::Vector3d> e);

// Vector3d <-> Vector3
void Eigen2Ros(const Eigen::Vector3d& e, geometry_msgs::Vector3& r);
void Ros2Eigen(const geometry_msgs::Vector3& r, Eigen::Ref<Eigen::Vector3d> e);

// Quateriond <-> Quaternion
void Eigen2Ros(const Eigen::Quaterniond& e, geometry_msgs::Quaternion& r);
void Ros2Eigen(const geometry_msgs::Quaternion& r, Eigen::Quaterniond& e);

// Vector3d, Quaterniond <-> Pose / Transform
void Eigen2Ros(const Eigen::Vector3d& pos,
               const Eigen::Quaterniond& quat,
               geometry_msgs::Pose& pose);
void Eigen2Ros(const Eigen::Vector3d& pos,
               const Eigen::Quaterniond& quat,
               geometry_msgs::Transform& tf);

// Isometry3d <-> Transform / Pose
void Ros2Eigen(const geometry_msgs::Transform& tf, Eigen::Isometry3d& iso);
void Eigen2Ros(const Eigen::Isometry3d& iso, geometry_msgs::Pose& pose);

void Sophus2Ros(const Sophus::SE3d& se3, geometry_msgs::Pose& pose);
void Sophus2Ros(const Sophus::SE3d& se3, geometry_msgs::Transform& tf);
void Sophus2Ros(const Sophus::SO3d& so3, geometry_msgs::Quaternion& quat);
void Ros2Sophus(const geometry_msgs::Quaternion& quat, Sophus::SO3d& so3);
void Ros2Sophus(const geometry_msgs::Pose& pose, Sophus::SE3d& se3);

void Ros2Ros(const geometry_msgs::Pose& pose, geometry_msgs::Transform& tf);

using PointFields = std::vector<sensor_msgs::PointField>;

PointFields MakePointFields(const std::string& fstr);
int GetPointStep(const PointFields& fields);

/// @brief A simple wrapper around PointCloud2
struct Cloud2Helper {
  sensor_msgs::PointCloud2 cloud;

  auto width() const noexcept { return cloud.width; }
  auto height() const noexcept { return cloud.height; }
  auto rows() const noexcept { return height(); }
  auto cols() const noexcept { return width(); }
  auto empty() const noexcept { return cloud.data.empty(); }

  Cloud2Helper() = default;
  Cloud2Helper(int rows, int cols, const PointFields& fields);
  Cloud2Helper(int rows, int cols, const std::string& fstring)
      : Cloud2Helper(rows, cols, MakePointFields(fstring)) {}
  explicit Cloud2Helper(const std::string& fstring)
      : Cloud2Helper(0, 0, fstring) {}

  void resize(int rows, int cols);

  template <typename T>
  T* PtrAt(int i) {
    return reinterpret_cast<T*>(cloud.data.data() + i * cloud.point_step);
  }

  template <typename T>
  T* PtrAt(int r, int c) {
    return PtrAt<T>(r * cloud.width + c);
  }
};

}  // namespace sv

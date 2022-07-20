#pragma once

#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>
#include <ros/node_handle.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>

#include "sv/dsol/direct.h"
#include "sv/dsol/odom.h"
#include "sv/dsol/select.h"
#include "sv/dsol/stereo.h"

namespace sv::dsol {

SelectCfg ReadSelectCfg(const ros::NodeHandle& pnh);
DirectCfg ReadDirectCfg(const ros::NodeHandle& pnh);
StereoCfg ReadStereoCfg(const ros::NodeHandle& pnh);
OdomCfg ReadOdomCfg(const ros::NodeHandle& pnh);

Camera MakeCamera(const sensor_msgs::CameraInfo& cinfo_msg);
void UpdateCamera(const sensor_msgs::CameraInfo& cinfo_msg, Camera& camera);

void Keyframe2Cloud(const Keyframe& kefyrame,
                    sensor_msgs::PointCloud2& cloud,
                    double max_depth,
                    int offset = 0);
void Keyframes2Cloud(const KeyframePtrConstSpan& keyframes,
                     sensor_msgs::PointCloud2& cloud,
                     double max_depth);

void DrawAlignGraph(const Eigen::Vector3d& frame_pos,
                    const Eigen::Matrix3Xd& kfs_pos,
                    const std::vector<int>& tracks,
                    const cv::Scalar& color,
                    double scale,
                    visualization_msgs::Marker& marker);

/// @brief Publish pose and also add it to path
struct PosePathPublisher {
  PosePathPublisher() = default;
  PosePathPublisher(ros::NodeHandle pnh,
                    const std::string& name,
                    const std::string& frame_id);

  geometry_msgs::PoseStamped Publish(const ros::Time& time,
                                     const Sophus::SE3d& tf);

  std::string frame_id_;
  ros::Publisher pose_pub_;
  ros::Publisher path_pub_;
  nav_msgs::Path path_msg_;
};

}  // namespace sv::dsol

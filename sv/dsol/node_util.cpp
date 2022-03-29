#include "sv/dsol/node_util.h"

#include "sv/ros1/msg_conv.h"
#include "sv/util/logging.h"

namespace sv::dsol {

namespace gm = geometry_msgs;
namespace vm = visualization_msgs;
static constexpr auto kNaNF = std::numeric_limits<float>::quiet_NaN();

SelectCfg ReadSelectCfg(const ros::NodeHandle& pnh) {
  SelectCfg cfg;
  pnh.getParam("sel_level", cfg.sel_level);
  pnh.getParam("cell_size", cfg.cell_size);
  pnh.getParam("min_grad", cfg.min_grad);
  pnh.getParam("max_grad", cfg.max_grad);
  pnh.getParam("nms_size", cfg.nms_size);
  pnh.getParam("min_ratio", cfg.min_ratio);
  pnh.getParam("max_ratio", cfg.max_ratio);
  pnh.getParam("reselect", cfg.reselect);
  return cfg;
}

DirectCfg ReadDirectCfg(const ros::NodeHandle& pnh) {
  DirectCfg cfg;

  pnh.getParam("init_level", cfg.optm.init_level);
  pnh.getParam("max_iters", cfg.optm.max_iters);
  pnh.getParam("max_xs", cfg.optm.max_xs);

  pnh.getParam("affine", cfg.cost.affine);
  pnh.getParam("stereo", cfg.cost.stereo);
  pnh.getParam("c2", cfg.cost.c2);
  pnh.getParam("dof", cfg.cost.dof);
  pnh.getParam("max_outliers", cfg.cost.max_outliers);
  pnh.getParam("grad_factor", cfg.cost.grad_factor);
  pnh.getParam("min_depth", cfg.cost.min_depth);

  return cfg;
}

StereoCfg ReadStereoCfg(const ros::NodeHandle& pnh) {
  StereoCfg cfg;
  pnh.getParam("half_rows", cfg.half_rows);
  pnh.getParam("half_cols", cfg.half_cols);
  pnh.getParam("match_level", cfg.match_level);
  pnh.getParam("refine_size", cfg.refine_size);
  pnh.getParam("min_zncc", cfg.min_zncc);
  pnh.getParam("min_depth", cfg.min_depth);
  return cfg;
}

OdomCfg ReadOdomCfg(const ros::NodeHandle& pnh) {
  OdomCfg cfg;
  pnh.getParam("marg", cfg.marg);
  pnh.getParam("num_kfs", cfg.num_kfs);
  pnh.getParam("num_levels", cfg.num_levels);
  pnh.getParam("min_track_ratio", cfg.min_track_ratio);
  pnh.getParam("vis_min_depth", cfg.vis_min_depth);

  pnh.getParam("reinit", cfg.reinit);
  pnh.getParam("init_depth", cfg.init_depth);
  pnh.getParam("init_stereo", cfg.init_stereo);
  pnh.getParam("init_align", cfg.init_align);
  return cfg;
}

Camera MakeCamera(const sensor_msgs::CameraInfo& cinfo_msg) {
  const cv::Size size(cinfo_msg.width, cinfo_msg.height);
  const auto& K = cinfo_msg.K;
  // K
  // 0, 1, 2
  // 3, 4, 5
  // 6, 7, 8
  Eigen::Array4d fc;
  fc << K[0], K[4], K[2], K[5];
  // P
  // 0, 1,  2,  3
  // 4, 5,  6,  7
  // 8, 9, 10, 11
  return {size, fc, cinfo_msg.P[3] / K[0]};
}

void Keyframe2Cloud(const Keyframe& keyframe,
                    sensor_msgs::PointCloud2& cloud,
                    double max_depth,
                    int offset) {
  const auto& points = keyframe.points();
  const auto& patches = keyframe.patches().front();
  const auto grid_size = points.cvsize();

  const auto total_size = offset + grid_size.area();
  cloud.data.resize(total_size * cloud.point_step);
  cloud.height = 1;
  cloud.width = total_size;

  for (int gr = 0; gr < points.rows(); ++gr) {
    for (int gc = 0; gc < points.cols(); ++gc) {
      const auto i = offset + gr * grid_size.width + gc;
      auto* ptr =
          reinterpret_cast<float*>(cloud.data.data() + i * cloud.point_step);

      const auto& point = points.at(gr, gc);
      // Only draw points with max info and within max depth
      if (!point.InfoMax() || (1.0 / point.idepth()) > max_depth) {
        ptr[0] = ptr[1] = ptr[2] = kNaNF;
        continue;
      }
      CHECK(point.PixelOk());
      CHECK(point.DepthOk());

      // transform to fixed frame
      const Eigen::Vector3f p_w = (keyframe.Twc() * point.pt()).cast<float>();
      const auto& patch = patches.at(gr, gc);

      ptr[0] = p_w.x();
      ptr[1] = p_w.y();
      ptr[2] = p_w.z();
      ptr[3] = static_cast<float>(patch.vals[0] / 255.0);
    }
  }
}

void Keyframes2Cloud(const KeyframePtrConstSpan& keyframes,
                     sensor_msgs::PointCloud2& cloud,
                     double max_depth) {
  if (keyframes.empty()) return;

  const auto num_kfs = static_cast<int>(keyframes.size());
  const auto grid_size = keyframes[0]->points().cvsize();

  // Set all points to bad
  const auto total_size = num_kfs * grid_size.area();
  cloud.data.reserve(total_size * cloud.point_step);
  cloud.height = 1;
  cloud.width = total_size;

  for (int k = 0; k < num_kfs; ++k) {
    Keyframe2Cloud(
        GetKfAt(keyframes, k), cloud, max_depth, grid_size.area() * k);
  }
}

/// ============================================================================
PosePathPublisher::PosePathPublisher(ros::NodeHandle pnh,
                                     const std::string& name,
                                     const std::string& frame_id)
    : frame_id_{frame_id},
      pose_pub_{pnh.advertise<gm::PoseStamped>("pose_" + name, 1)},
      path_pub_{pnh.advertise<nav_msgs::Path>("path_" + name, 1)} {
  path_msg_.poses.reserve(1024);
}

gm::PoseStamped PosePathPublisher::Publish(const ros::Time& time,
                                           const Sophus::SE3d& tf) {
  gm::PoseStamped pose_msg;
  pose_msg.header.stamp = time;
  pose_msg.header.frame_id = frame_id_;
  Sophus2Ros(tf, pose_msg.pose);
  pose_pub_.publish(pose_msg);

  path_msg_.header = pose_msg.header;
  path_msg_.poses.push_back(pose_msg);
  path_pub_.publish(path_msg_);
  return pose_msg;
}

void DrawAlignGraph(const Eigen::Vector3d& frame_pos,
                    const Eigen::Matrix3Xd& kfs_pos,
                    const std::vector<int>& tracks,
                    const cv::Scalar& color,
                    double scale,
                    vm::Marker& marker) {
  CHECK_EQ(tracks.size(), kfs_pos.cols());
  marker.ns = "align";
  marker.id = 0;
  marker.type = vm::Marker::LINE_LIST;
  marker.action = vm::Marker::ADD;
  marker.color.b = color[0];
  marker.color.g = color[1];
  marker.color.r = color[2];
  marker.color.a = 1.0;

  marker.scale.x = scale;
  marker.pose.orientation.w = 1.0;
  const auto num_kfs = tracks.size();
  marker.points.clear();
  marker.points.reserve(num_kfs * 2);

  gm::Point p0;
  p0.x = frame_pos.x();
  p0.y = frame_pos.y();
  p0.z = frame_pos.z();

  gm::Point p1;
  for (int i = 0; i < num_kfs; ++i) {
    if (tracks[i] <= 0) continue;
    p1.x = kfs_pos.col(i).x();
    p1.y = kfs_pos.col(i).y();
    p1.z = kfs_pos.col(i).z();
    marker.points.push_back(p0);
    marker.points.push_back(p1);
  }
}

}  // namespace sv::dsol

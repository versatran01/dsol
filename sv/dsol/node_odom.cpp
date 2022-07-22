#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseArray.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <tf2_ros/transform_broadcaster.h>

#include <boost/circular_buffer.hpp>

#include "sv/dsol/extra.h"
#include "sv/dsol/node_util.h"
#include "sv/dsol/odom.h"
#include "sv/ros1/msg_conv.h"

namespace sv::dsol {

namespace cb = cv_bridge;
namespace sm = sensor_msgs;
namespace gm = geometry_msgs;
namespace mf = message_filters;

struct NodeOdom {
  explicit NodeOdom(const ros::NodeHandle& pnh);

  void InitOdom();
  void InitRosIO();

  void Cinfo1Cb(const sm::CameraInfo& cinfo1_msg);
  void StereoCb(const sm::ImageConstPtr& image0_ptr,
                const sm::ImageConstPtr& image1_ptr);

  void TfCamCb(const gm::Transform& tf_cam_msg);
  void TfImuCb(const gm::Transform& tf_imu_msg);

  void AccCb(const sm::Imu& acc_msg);
  void GyrCb(const sm::Imu& gyr_msg);

  void PublishOdom(const std_msgs::Header& header, const Sophus::SE3d& tf);
  void PublishCloud(const std_msgs::Header& header);

  using StereoSync = mf::TimeSynchronizer<sm::Image, sm::Image>;

  ros::NodeHandle pnh_;

  boost::circular_buffer<sm::Imu> gyrs_;
  mf::Subscriber<sm::Image> sub_image0_;
  mf::Subscriber<sm::Image> sub_image1_;
  StereoSync sync_stereo_;

  ros::Subscriber sub_cinfo1_;
  //  ros::Subscriber sub_acc_;
  ros::Subscriber sub_gyr_;

  ros::Publisher pub_points_;
  ros::Publisher pub_parray_;
  PosePathPublisher pub_odom_;

  MotionModel motion_;
  DirectOdometry odom_;

  std::string frame_{"fixed"};
  sm::PointCloud2 cloud_;
};

NodeOdom::NodeOdom(const ros::NodeHandle& pnh)
    : pnh_(pnh),
      gyrs_(50),
      sub_image0_(pnh_, "image0", 5),
      sub_image1_(pnh_, "image1", 5),
      sync_stereo_(sub_image0_, sub_image1_, 5) {
  InitOdom();
  InitRosIO();
}

void NodeOdom::InitOdom() {
  {
    auto cfg = ReadOdomCfg({pnh_, "odom"});
    pnh_.getParam("tbb", cfg.tbb);
    pnh_.getParam("log", cfg.log);
    pnh_.getParam("vis", cfg.vis);
    odom_.Init(cfg);
  }
  odom_.selector = PixelSelector(ReadSelectCfg({pnh_, "select"}));
  odom_.matcher = StereoMatcher(ReadStereoCfg({pnh_, "stereo"}));
  odom_.aligner = FrameAligner(ReadDirectCfg({pnh_, "align"}));
  odom_.adjuster = BundleAdjuster(ReadDirectCfg({pnh_, "adjust"}));
  odom_.cmap = GetColorMap(pnh_.param<std::string>("cm", "jet"));
  ROS_INFO_STREAM(odom_.Repr());

  // Init motion model
  motion_.Init();
}

void NodeOdom::InitRosIO() {
  sync_stereo_.registerCallback(boost::bind(&NodeOdom::StereoCb, this, _1, _2));
  sub_cinfo1_ = pnh_.subscribe("cinfo1", 1, &NodeOdom::Cinfo1Cb, this);
  sub_gyr_ = pnh_.subscribe("gyr", 200, &NodeOdom::GyrCb, this);
  // sub_acc_ = pnh_.subscribe("acc", 100, &NodeOdom::AccCb, this);

  pub_odom_ = PosePathPublisher(pnh_, "odom", frame_);
  pub_points_ = pnh_.advertise<sm::PointCloud2>("points", 1);
  pub_parray_ = pnh_.advertise<gm::PoseArray>("parray", 1);
}

void NodeOdom::Cinfo1Cb(const sensor_msgs::CameraInfo& cinfo1_msg) {
  odom_.camera = MakeCamera(cinfo1_msg);
  ROS_INFO_STREAM(odom_.camera.Repr());
  sub_cinfo1_.shutdown();
}

void NodeOdom::AccCb(const sensor_msgs::Imu& acc_msg) {}

void NodeOdom::GyrCb(const sensor_msgs::Imu& gyr_msg) {
  // Normally there is a transform from imu to camera, but in realsense, imu and
  // left infrared camera are aligned (only small translation, so we skip
  // reading the tf)

  gyrs_.push_back(gyr_msg);
}

void NodeOdom::StereoCb(const sensor_msgs::ImageConstPtr& image0_ptr,
                        const sensor_msgs::ImageConstPtr& image1_ptr) {
  const auto curr_header = image0_ptr->header;
  const auto image0 = cb::toCvShare(image0_ptr)->image;
  const auto image1 = cb::toCvShare(image1_ptr)->image;

  // Get delta time
  static ros::Time prev_stamp;
  const auto delta_duration =
      prev_stamp.isZero() ? ros::Duration{} : curr_header.stamp - prev_stamp;
  const auto dt = delta_duration.toSec();
  ROS_INFO_STREAM("dt: " << dt * 1000);

  // Motion model
  Sophus::SE3d dtf_pred;
  if (dt > 0) {
    // Do a const vel prediction first
    dtf_pred = motion_.PredictDelta(dt);

    // Then overwrite rotation part if we have imu
    // TODO(dsol): Use 0th order integration, maybe switch to 1st order later
    ROS_INFO_STREAM(
        fmt::format("prev: {}, curr: {}, first_imu: {}, last_imu: {}",
                    prev_stamp.toSec(),
                    curr_header.stamp.toSec(),
                    gyrs_.front().header.stamp.toSec(),
                    gyrs_.back().header.stamp.toSec()));
    Sophus::SO3d dR{};
    int n_imus = 0;
    for (size_t i = 0; i < gyrs_.size(); ++i) {
      const auto& imu = gyrs_[i];
      // Skip imu msg that is earlier than the previous odom
      if (imu.header.stamp <= prev_stamp) continue;
      if (imu.header.stamp > curr_header.stamp) continue;

      const auto prev_imu_stamp =
          i == 0 ? prev_stamp : gyrs_.at(i - 1).header.stamp;
      const double dt_imu = (imu.header.stamp - prev_imu_stamp).toSec();
      CHECK_GT(dt_imu, 0);
      Eigen::Map<const Eigen::Vector3d> w(&imu.angular_velocity.x);
      dR *= Sophus::SO3d::exp(w * dt_imu);
      ++n_imus;
    }
    ROS_INFO_STREAM("n_imus: " << n_imus);
    // We just replace const vel prediction
    if (n_imus > 0) dtf_pred.so3() = dR;
  }

  const auto status = odom_.Estimate(image0, image1, dtf_pred);
  ROS_INFO_STREAM(status.Repr());

  // Motion model correct if tracking is ok and not first frame
  if (status.track.ok) {
    motion_.Correct(status.Twc(), dt);
  } else {
    ROS_WARN_STREAM("Tracking failed (or 1st frame), slow motion model");
  }

  // publish stuff
  std_msgs::Header header;
  header.frame_id = "fixed";
  header.stamp = curr_header.stamp;

  PublishOdom(header, status.Twc());
  if (status.map.remove_kf) {
    PublishCloud(header);
  }

  prev_stamp = curr_header.stamp;
}

void NodeOdom::PublishOdom(const std_msgs::Header& header,
                           const Sophus::SE3d& tf) {
  // Publish odom poses
  const auto pose_msg = pub_odom_.Publish(header.stamp, tf);

  // Publish keyframe poses
  const auto poses = odom_.window.GetAllPoses();
  gm::PoseArray parray_msg;
  parray_msg.header = header;
  parray_msg.poses.resize(poses.size());
  for (size_t i = 0; i < poses.size(); ++i) {
    Sophus2Ros(poses.at(i), parray_msg.poses.at(i));
  }
  pub_parray_.publish(parray_msg);
}

void NodeOdom::PublishCloud(const std_msgs::Header& header) {
  if (pub_points_.getNumSubscribers() == 0) return;

  cloud_.header = header;
  cloud_.point_step = 16;
  cloud_.fields = MakePointFields("xyzi");

  ROS_DEBUG_STREAM(odom_.window.MargKf().status().Repr());
  Keyframe2Cloud(odom_.window.MargKf(), cloud_, 50.0);
  pub_points_.publish(cloud_);
}

// void NodeOdom::TfCamCb(const geometry_msgs::Transform& tf_cam_msg) {
//   odom_.camera.baseline_ = -tf_cam_msg.translation.x;
//   ROS_INFO_STREAM(odom_.camera.Repr());
// }

// void NodeOdom::TfImuCb(const geometry_msgs::Transform& tf_imu_msg) {}

}  // namespace sv::dsol

int main(int argc, char** argv) {
  ros::init(argc, argv, "dsol_odom");
  cv::setNumThreads(4);
  sv::dsol::NodeOdom node{ros::NodeHandle{"~"}};
  ros::spin();
}

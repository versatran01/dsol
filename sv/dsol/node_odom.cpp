#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseArray.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <tf2_ros/transform_broadcaster.h>

#include "sv/dsol/extra.h"
#include "sv/dsol/node_util.h"
#include "sv/dsol/odom.h"
#include "sv/ros1/msg_conv.h"

namespace sv::dsol {

namespace cb = cv_bridge;
namespace sm = sensor_msgs;
namespace gm = geometry_msgs;

struct NodeOdom {
  explicit NodeOdom(const ros::NodeHandle& pnh);

  void InitOdom();
  void InitRosIO();

  void Image0Cb(const sm::ImageConstPtr& image0_ptr);
  void Image1Cb(const sm::ImageConstPtr& image1_ptr);
  void Cinfo0Cb(const sm::CameraInfo& cinfo0_msg);

  void TfCamCb(const gm::Transform& tf_cam_msg);
  void TfImuCb(const gm::Transform& tf_imu_msg);

  void AccCb(const sm::Imu& acc_msg);
  void GyrCb(const sm::Imu& gyr_msg);

  void Estimate();
  void PublishOdom(const std_msgs::Header& header, const Sophus::SE3d& tf);
  void PublishCloud(const std_msgs::Header& header);

  ros::NodeHandle pnh_;

  ros::Subscriber sub_cinfo0_;
  ros::Subscriber sub_image0_;
  ros::Subscriber sub_image1_;
  ros::Subscriber sub_tf_cam_;
  ros::Subscriber sub_tf_imu_;
  ros::Subscriber sub_acc_;
  ros::Subscriber sub_gyr_;

  ros::Publisher pub_points_;
  ros::Publisher pub_parray_;
  PosePathPublisher pub_odom_;

  sm::ImageConstPtr image0_ptr_;
  sm::ImageConstPtr image1_ptr_;

  MotionModel motion_;
  DirectOdometry odom_;

  std::string frame_{"fixed"};
  sm::PointCloud2 cloud_;
};

NodeOdom::NodeOdom(const ros::NodeHandle& pnh) : pnh_{pnh} {
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
  sub_image0_ = pnh_.subscribe("image0", 5, &NodeOdom::Image0Cb, this);
  sub_image1_ = pnh_.subscribe("image1", 5, &NodeOdom::Image1Cb, this);
  sub_cinfo0_ = pnh_.subscribe("cinfo0", 1, &NodeOdom::Cinfo0Cb, this);
  sub_tf_cam_ = pnh_.subscribe("tf_cam", 1, &NodeOdom::TfCamCb, this);
  sub_tf_imu_ = pnh_.subscribe("tf_imu", 1, &NodeOdom::TfImuCb, this);
  sub_acc_ = pnh_.subscribe("acc", 100, &NodeOdom::AccCb, this);
  sub_gyr_ = pnh_.subscribe("gyr", 100, &NodeOdom::GyrCb, this);

  pub_odom_ = PosePathPublisher(pnh_, "odom", frame_);
  pub_points_ = pnh_.advertise<sm::PointCloud2>("points", 1);
  pub_parray_ = pnh_.advertise<gm::PoseArray>("parray", 1);
}

void NodeOdom::Cinfo0Cb(const sensor_msgs::CameraInfo& cinfo0_msg) {
  UpdateCamera(cinfo0_msg, odom_.camera);
  ROS_INFO_STREAM(odom_.camera.Repr());
}

void NodeOdom::Image0Cb(const sensor_msgs::ImageConstPtr& image0_ptr) {
  if (!odom_.camera.Ok() || !odom_.camera.is_stereo()) return;

  // save image
  image0_ptr_ = image0_ptr;
  if (!image1_ptr_) return;
  Estimate();
}

void NodeOdom::Image1Cb(const sensor_msgs::ImageConstPtr& image1_ptr) {
  if (!odom_.camera.Ok() || !odom_.camera.is_stereo()) return;

  // save image
  image1_ptr_ = image1_ptr;
  if (!image0_ptr_) return;
  Estimate();
}

void NodeOdom::AccCb(const sensor_msgs::Imu& acc_msg) {}

void NodeOdom::GyrCb(const sensor_msgs::Imu& gyr_msg) {}

void NodeOdom::Estimate() {
  CHECK_NOTNULL(image0_ptr_);
  CHECK_NOTNULL(image1_ptr_);
  if (image0_ptr_->header.stamp != image1_ptr_->header.stamp) {
    ROS_WARN_STREAM(fmt::format("image0 and image1 stamp not equal: {} vs {}",
                                image0_ptr_->header.stamp,
                                image1_ptr_->header.stamp));
    image0_ptr_.reset();
    image1_ptr_.reset();
    return;
  }

  const auto image_header = image0_ptr_->header;
  const auto image0 = cb::toCvShare(image0_ptr_)->image;
  const auto image1 = cb::toCvShare(image1_ptr_)->image;

  // Get delta time
  static ros::Time prev_stamp;
  const auto delta_duration =
      prev_stamp.isZero() ? ros::Duration{} : image_header.stamp - prev_stamp;
  const auto dt = delta_duration.toSec();
  ROS_INFO_STREAM("dt: " << dt * 1000);

  // Motion model
  const auto dtf_pred = motion_.PredictDelta(dt);
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
  header.stamp = image_header.stamp;

  PublishOdom(header, status.Twc());
  if (status.map.remove_kf) {
    PublishCloud(header);
  }

  prev_stamp = image_header.stamp;
  image0_ptr_.reset();
  image1_ptr_.reset();
}

void NodeOdom::PublishOdom(const std_msgs::Header& header,
                           const Sophus::SE3d& tf) {
  const auto pose_msg = pub_odom_.Publish(header.stamp, tf);

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

void NodeOdom::TfCamCb(const geometry_msgs::Transform& tf_cam_msg) {
  odom_.camera.baseline_ = -tf_cam_msg.translation.x;
  ROS_INFO_STREAM(odom_.camera.Repr());
}

void NodeOdom::TfImuCb(const geometry_msgs::Transform& tf_imu_msg) {}

}  // namespace sv::dsol

int main(int argc, char** argv) {
  ros::init(argc, argv, "dsol_odom");
  sv::dsol::NodeOdom node{ros::NodeHandle{"~"}};
  ros::spin();
}

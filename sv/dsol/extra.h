#pragma once

#include <fstream>

#include "sv/dsol/frame.h"
#include "sv/dsol/select.h"
#include "sv/util/dataset.h"
#include "sv/util/summary.h"

namespace sv::dsol {

struct MotionModel {
  MotionModel() = default;
  explicit MotionModel(double alpha) : alpha_{alpha} {}

  bool init_{false};
  double alpha_{0.5};  // v1' <- v0 * (1-a) + v1 * a;
  Sophus::SE3d T_last_{};
  Eigen::Vector3d omg_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d vel_{Eigen::Vector3d::Zero()};

  double alpha() const noexcept { return alpha_; }

  bool Ok() const noexcept { return init_; }
  void Init(const Sophus::SE3d& T_w_c = {},
            const Eigen::Vector3d& vel = {0, 0, 0},
            const Eigen::Vector3d& omg = {0, 0, 0});

  Sophus::SE3d Predict(double dt) const { return T_last_ * PredictDelta(dt); }

  Sophus::SE3d PredictDelta(double dt) const {
    return {Sophus::SO3d::exp(omg_ * dt), vel_ * dt};
  }

  void Correct(const Sophus::SE3d& T_w_c, double dt);

  void Scale(double s) {
    omg_.array() *= s;
    vel_.array() *= s;
  }

  void ResetVelocity() noexcept {
    vel_.setZero();
    omg_.setZero();
  }
};

/// @brief Write poses to file in TUM format
/// https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
/// timestamp tx ty tz qx qy qz qw
class TumFormatWriter {
  std::string filename_{};
  std::ofstream ofs_{};

 public:
  /// @brief If filename is empty, it will not write to anything
  explicit TumFormatWriter(const std::string& filename = "");

  const std::string& filename() const noexcept { return filename_; }
  bool IsDummy() const noexcept { return filename_.empty(); }

  /// @brief Write to kitti
  void Write(int64_t i, const Sophus::SE3d& pose);
};

struct PlayCfg {
  int index{};
  int nframes{};
  int skip{};
  int nlevels{};
  bool affine{};

  void Check() const;
  std::string Repr() const;
};

/// @brief
struct PlayData {
  PlayData() = default;
  PlayData(const Dataset& dataset, const PlayCfg& cfg);

  bool empty() const noexcept { return frames.empty(); }
  auto size() const noexcept { return frames.size(); }

  Camera camera;
  std::vector<Frame> frames;
  std::vector<cv::Mat> depths;
  std::vector<Sophus::SE3d> poses;
};

void InitKfWithDepth(Keyframe& kf,
                     const Camera& camera,
                     PixelSelector& selector,
                     const cv::Mat& depth,
                     TimerSummary& tm,
                     int gsize = 0);

}  // namespace sv::dsol

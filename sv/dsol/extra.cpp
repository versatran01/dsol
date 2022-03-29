#include "sv/dsol/extra.h"

#include <opencv2/imgproc.hpp>

#include "sv/util/logging.h"

namespace sv::dsol {

void MotionModel::Init(const Sophus::SE3d& T_w_c,
                       const Eigen::Vector3d& vel,
                       const Eigen::Vector3d& omg) {
  T_last_ = T_w_c;
  vel_ = vel;
  omg_ = omg;
  init_ = true;
}

void MotionModel::Correct(const Sophus::SE3d& T_w_c, double dt) {
  const auto tf_delta = T_last_.inverse() * T_w_c;
  omg_ = (1 - alpha_) * omg_ + (alpha_ / dt) * tf_delta.so3().log();
  vel_ = (1 - alpha_) * vel_ + (alpha_ / dt) * tf_delta.translation();
  T_last_ = T_w_c;
}

/// ============================================================================
TumFormatWriter::TumFormatWriter(const std::string& filename)
    : filename_{filename} {
  if (!filename_.empty()) {
    ofs_ = std::ofstream{filename_};
  }
}

void TumFormatWriter::Write(int64_t i, const Sophus::SE3d& pose) {
  if (!ofs_.good()) return;

  const auto& t = pose.translation();
  const auto& q = pose.unit_quaternion();
  const auto line = fmt::format("{} {} {} {} {} {} {} {}",
                                i,
                                t.x(),
                                t.y(),
                                t.z(),
                                q.x(),
                                q.y(),
                                q.z(),
                                q.w());
  ofs_ << line << std::endl;
}

/// ============================================================================
PlayData::PlayData(const Dataset& dataset, const PlayCfg& cfg)
    : frames(cfg.nframes), depths(cfg.nframes), poses(cfg.nframes) {
  const auto tf_c0_w = SE3dFromMat(dataset.Get(DataType::kPose, 0)).inverse();

  for (int k = 0; k < cfg.nframes; ++k) {
    const int i = cfg.index + k * (cfg.skip + 1);
    LOG(INFO) << "Reading index: " << i;

    // pose
    const auto pose = dataset.Get(DataType::kPose, i);
    const auto tf_w_c = SE3dFromMat(pose);
    const auto tf_c0_c = tf_c0_w * tf_w_c;

    int aff_val = cfg.affine ? k : 0;

    // stereo images
    ImagePyramid grays_l;
    {
      auto image = dataset.Get(DataType::kImage, i, 0);
      if (image.type() == CV_8UC3) {
        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
      }

      if (aff_val != 0) image += aff_val;
      MakeImagePyramid(image, cfg.nlevels, grays_l);
    }

    ImagePyramid grays_r;
    {
      auto image = dataset.Get(DataType::kImage, i, 1);
      if (image.type() == CV_8UC3) {
        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
      }
      if (aff_val != 0) image += aff_val;
      MakeImagePyramid(image, cfg.nlevels, grays_r);
    }

    AffineModel affm{0, static_cast<double>(aff_val)};
    frames.at(k) = Frame(grays_l, grays_r, tf_c0_c, affm, affm);
    depths.at(k) = dataset.Get(DataType::kDepth, i);
    poses.at(k) = tf_c0_c;  // gt

    LOG(INFO) << fmt::format("frame {}: {}", k, frames.at(k));
  }

  const auto intrin = dataset.Get(DataType::kIntrin, 0);
  camera = Camera::FromMat(frames.front().cvsize(), intrin);
}

void InitKfWithDepth(Keyframe& kf,
                     const Camera& camera,
                     PixelSelector& selector,
                     const cv::Mat& depth,
                     TimerSummary& tm,
                     int gsize) {
  {
    auto t = tm.Scoped("SelectPixels");
    selector.Select(kf.grays_l(), gsize);
  }

  {
    auto t = tm.Scoped("InitPoints");
    kf.InitPoints(selector.pixels(), camera);
  }

  {
    auto t = tm.Scoped("InitDepths");
    kf.InitFromDepth(depth);
  }

  {
    auto t = tm.Scoped("InitPatches");
    kf.InitPatches(gsize);
  }

  kf.UpdateStatusInfo();
}

void PlayCfg::Check() const {
  CHECK_GE(index, 0);
  CHECK_GT(nframes, 0);
  CHECK_GE(skip, 0);
  CHECK_GT(nlevels, 0);
}

std::string PlayCfg::Repr() const {
  return fmt::format(
      "PlayCfg(index={}, nframes={}, skip={}, nlevels={}, affine={})",
      index,
      nframes,
      skip,
      nlevels,
      affine);
}

}  // namespace sv::dsol

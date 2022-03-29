#include <absl/flags/flag.h>
#include <absl/flags/parse.h>

#include "sv/dsol/extra.h"
#include "sv/dsol/odom.h"
#include "sv/util/cmap.h"
#include "sv/util/dataset.h"
#include "sv/util/logging.h"
#include "sv/util/ocv.h"
#include "sv/util/summary.h"

ABSL_FLAG(bool, tbb, false, "use tbb");
ABSL_FLAG(bool, vis, false, "visualize");
ABSL_FLAG(int32_t, wait, 0, "wait ms");
ABSL_FLAG(int32_t, ind, 0, "dataset index");
ABSL_FLAG(int32_t, len, 0, "length");
ABSL_FLAG(std::string,
          dir,
          "/home/chao/Workspace/dataset/vkitti/Scene01/clone",
          "dataset dir");

ABSL_FLAG(int32_t, num_levels, 5, "number of levels");
ABSL_FLAG(int32_t, num_kfs, 4, "number of kfs");

namespace sv::dsol {

Dataset dataset;
DirectOdometry odom;
Sophus::SE3d T_c0_w;

void Init() {
  dataset = Vkitti2(absl::GetFlag(FLAGS_dir));
  LOG(INFO) << dataset;

  T_c0_w = SE3dFromMat(dataset.Get(DataType::kPose, 0)).inverse();

  {
    OdomCfg cfg;
    cfg.tbb = static_cast<int>(absl::GetFlag(FLAGS_tbb));
    cfg.vis = absl::GetFlag(FLAGS_vis);
    odom.Init(cfg);
  }

  {
    const auto image = dataset.Get(DataType::kImage, 0);
    const auto intrin = dataset.Get(DataType::kIntrin, 0);
    odom.camera = Camera::FromMat({image.cols, image.rows}, intrin);
  }

  {
    const int num_kfs = absl::GetFlag(FLAGS_num_kfs);
    odom.window.Resize(num_kfs);
  }

  {
    SelectCfg cfg;
    odom.selector = PixelSelector(cfg);
  }
  {
    StereoCfg cfg;
    odom.matcher = StereoMatcher(cfg);
  }
  {
    AlignCfg cfg;
    odom.aligner = FrameAligner(cfg);
  }
  {
    AdjustCfg cfg;
    odom.adjuster = BundleAdjuster(cfg);
  }

  LOG(INFO) << odom;
}

void Run() {
  const double dt = 0.1;
  LOG(INFO) << "dt: " << dt;

  const int wait = absl::GetFlag(FLAGS_wait);

  const int ind = absl::GetFlag(FLAGS_ind);
  LOG(INFO) << "ind: " << ind;

  const int len = absl::GetFlag(FLAGS_len);
  LOG(INFO) << "len: " << len;

  const int end =
      len > 0 ? std::min(dataset.size(), ind + len) : dataset.size();
  LOG(INFO) << "end: " << end;

  const int num_levels = absl::GetFlag(FLAGS_num_levels);
  LOG(INFO) << "num_levels: " << num_levels;

  MotionModel motion(0.4);
  TumFormatWriter writer("/tmp/result.txt");

  for (int i = ind; i < end; ++i) {
    bool first_frame = i == ind;

    LOG(INFO) << fmt::format(fmt::fg(fmt::color::red), "ind: {}", i);

    ImagePyramid grays_left;
    {
      auto image = dataset.Get(DataType::kImage, i, 0);
      cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
      MakeImagePyramid(image, num_levels, grays_left);
    }

    ImagePyramid grays_right;
    {
      auto image = dataset.Get(DataType::kImage, i, 1);
      cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
      MakeImagePyramid(image, num_levels, grays_right);
    }

    const auto depth = dataset.Get(DataType::kDepth, i);

    const auto pose = dataset.Get(DataType::kPose, i);
    const auto T_w_c = SE3dFromMat(pose);
    const auto T_c0_c = T_c0_w * T_w_c;

    Sophus::SE3d dT_pred;
    if (!motion.Ok()) {
      motion.Init(T_c0_c);
      LOG(INFO) << "Initialize motion model:\n" << T_c0_c.matrix3x4();
    } else {
      dT_pred = motion.PredictDelta(dt);
    }

    LOG(INFO) << "trans pred:\t" << dT_pred.translation().transpose();
    LOG(INFO) << "trans gt:\t" << T_c0_c.translation().transpose();
    //    odom.Process(grays_left, grays_right, depth, T_pred);

    if (!first_frame) {
      motion.Correct(T_c0_c, dt);
    }

    //    writer.Write(i, odom.Twc());

    //    const auto trans_err =
    //        (odom.Twc().translation() - T_c0_c.translation()).eval();
    //    LOG(INFO) << "trans err:\t" << trans_err.transpose();
    //    LOG(INFO) << fmt::format(
    //        fmt::fg(fmt::color::green), "trans err norm:\t{}",
    //        trans_err.norm());

    cv::waitKey(wait);
  }
}

}  // namespace sv::dsol

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  sv::dsol::Init();
  sv::dsol::Run();
}

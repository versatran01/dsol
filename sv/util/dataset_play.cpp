#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <glog/logging.h>

#include "sv/util/dataset.h"
#include "sv/util/ocv.h"

ABSL_FLAG(std::string, data, "vk2", "dataset name (vk2, tta, kit)");
ABSL_FLAG(std::string,
          dir,
          // "/home/chao/Workspace/dataset/vkitti/Scene01/clone",
          // "/home/chao/Workspace/dataset/kitti/dataset/sequences/00",
          // "/home/chao/Workspace/dataset/tartan_air/office/Easy/P000",
          "/home/chao/Workspace/dataset/realsense/outdoor_slow",
          "dataset root dir");
ABSL_FLAG(double, idepth_scale, 4.0, "inverse depth scale");

using namespace sv;

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  const auto ds = CreateDataset(absl::GetFlag(FLAGS_dir));

  if (ds.Ok()) {
    LOG(INFO) << ds;
  } else {
    LOG(WARNING) << "Dataset not initialized";
    return 0;
  }

  const auto idepth_scale = absl::GetFlag(FLAGS_idepth_scale);

  const auto intrin = ds.Get(DataType::kIntrin, 0, 0);
  LOG(INFO) << intrin;

  WindowTiler tiler({1440, 900}, {400, 400}, {400, 0});

  for (int i = 0; i < ds.size(); ++i) {
    const auto image_left = ds.Get(DataType::kImage, i, 0);
    const auto depth_left = ds.Get(DataType::kDepth, i, 0);

    auto pose = ds.Get(DataType::kPose, i, 0);
    if (!pose.empty()) {
      pose = pose.reshape(4, 4);
      LOG(INFO) << "\n" << pose;
    }

    Imshow("image_left", image_left);
    if (!depth_left.empty()) {
      tiler.Tile("depth_left",
                 ApplyCmap(idepth_scale / depth_left, 1.0, cv::COLORMAP_PINK));
    }

    if (ds.is_stereo()) {
      const auto image_right = ds.Get(DataType::kImage, i, 1);
      if (!image_right.empty()) {
        tiler.Tile("image_right", image_right);
      }

      const auto depth_right = ds.Get(DataType::kDepth, i, 1);

      if (!depth_right.empty()) {
        tiler.Tile(
            "depth_right",
            ApplyCmap(idepth_scale / depth_right, 1.0, cv::COLORMAP_PINK));
      }
    }

    const int wait_ws = i == 0 ? -1 : 1;
    tiler.Reset();
    cv::waitKey(wait_ws);
  }
}

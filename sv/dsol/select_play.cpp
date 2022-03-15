#include <absl/flags/flag.h>
#include <absl/flags/parse.h>

#include "sv/dsol/extra.h"
#include "sv/dsol/select.h"
#include "sv/dsol/viz.h"
#include "sv/util/logging.h"
#include "sv/util/ocv.h"

ABSL_FLAG(bool, tbb, false, "use tbb");
ABSL_FLAG(bool, vis, true, "visualization");
ABSL_FLAG(int32_t, index, 0, "dataset index");
ABSL_FLAG(std::string,
          dir,
//          "/home/chao/Workspace/dataset/kitti/Scene20/clone",
//          "/media/chao/External/dataset/kitti/dataset/sequences/09",
          "/media/chao/External/dataset/tartan_air/carwelding/Easy/P002",
          "dataset dir");

ABSL_FLAG(int32_t, cell_size, 16, "cell size");
ABSL_FLAG(int32_t, sel_level, 1, "select level");
ABSL_FLAG(int32_t, min_grad, 8, "minimum gradient");
ABSL_FLAG(int32_t, max_grad, 64, "maximum gradient");
ABSL_FLAG(double, min_ratio, 0.0, "minimum ratio");
ABSL_FLAG(double, max_ratio, 1.0, "maximum ratio");
ABSL_FLAG(bool, reselect, true, "reselect if ratio too low");

ABSL_FLAG(int32_t, num_kfs, 3, "num keyframes");
ABSL_FLAG(int32_t, num_levels, 4, "num pyramid levels");

namespace sv::dsol {

void Run() {
  TimerSummary tm{"dsol"};

  const int tbb = static_cast<int>(absl::GetFlag(FLAGS_tbb));
  LOG(INFO) << "tbb: " << tbb;
  const bool vis = absl::GetFlag(FLAGS_vis);
  LOG(INFO) << "vis: " << vis;
  const auto dataset = CreateDataset(absl::GetFlag(FLAGS_dir));
  CHECK(dataset.Ok());
  LOG(INFO) << dataset;

  const int index = absl::GetFlag(FLAGS_index);
  LOG(INFO) << "index: " << index;
  const int nlevels = absl::GetFlag(FLAGS_num_levels);
  LOG(INFO) << "num_levels: " << nlevels;
  const int nkfs = absl::GetFlag(FLAGS_num_kfs);
  LOG(INFO) << "num_kfs: " << nkfs;

  PlayData data(dataset, index, nkfs, nlevels, false);

  SelectCfg sel_cfg;
  sel_cfg.sel_level = absl::GetFlag(FLAGS_sel_level);
  sel_cfg.cell_size = absl::GetFlag(FLAGS_cell_size);
  sel_cfg.min_grad = absl::GetFlag(FLAGS_min_grad);
  sel_cfg.max_grad = absl::GetFlag(FLAGS_max_grad);
  sel_cfg.min_ratio = absl::GetFlag(FLAGS_min_ratio);
  sel_cfg.max_ratio = absl::GetFlag(FLAGS_max_ratio);
  sel_cfg.reselect = absl::GetFlag(FLAGS_reselect);
  PixelSelector selector{sel_cfg};
  LOG(INFO) << selector.Repr();

  WindowTiler tiler;

  for (int k = 0; k < nkfs; ++k) {
    const auto& frame = data.frames.at(k);

    int npixels = 0;
    {
      auto t = tm.Scoped("SelectPixels");
      npixels = selector.Select(frame.grays_l(), tbb);
    }
    LOG(INFO) << fmt::format("{}: n pixels {}", k, npixels);

    if (vis) {
      const auto color = CV_RGB(0, 255, 255);
      cv::Mat disp;
      cv::cvtColor(frame.grays_l().front(), disp, cv::COLOR_GRAY2BGR);
      DrawSelectedPixels(disp, selector.pixels(), color, 1);
      tiler.Tile(std::to_string(k), disp);
    }
  }

  LOG(INFO) << tm.ReportAll(true);

  cv::waitKey(-1);
}

}  // namespace sv::dsol

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  sv::dsol::Run();
}

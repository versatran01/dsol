#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <fmt/ranges.h>

#include "sv/dsol/extra.h"
#include "sv/dsol/select.h"
#include "sv/dsol/stereo.h"
#include "sv/dsol/viz.h"
#include "sv/util/cmap.h"
#include "sv/util/metric.h"
#include "sv/util/ocv.h"

ABSL_FLAG(bool, tbb, false, "use tbb");
ABSL_FLAG(bool, vis, true, "visualization");
ABSL_FLAG(int32_t, index, 0, "dataset index");
ABSL_FLAG(std::string,
          dir,
          // "/home/chao/Workspace/dataset/vkitti/Scene01/clone",
          "/home/chao/Workspace/dataset/tartan_air/office2/Easy/P000",
          "dataset dir");

ABSL_FLAG(int32_t, num_kfs, 1, "num keyframes");
ABSL_FLAG(int32_t, num_levels, 4, "num pyramid levels");

ABSL_FLAG(int32_t, cell_size, 16, "cell size");
ABSL_FLAG(int32_t, sel_level, 1, "select level");
ABSL_FLAG(int32_t, min_grad, 8, "minimum gradient");
ABSL_FLAG(int32_t, max_grad, 64, "maximum gradient");
ABSL_FLAG(double, min_ratio, 0.0, "minimum ratio");
ABSL_FLAG(double, max_ratio, 1.0, "maximum ratio");
ABSL_FLAG(bool, reselect, true, "reselect if ratio too low");

ABSL_FLAG(int32_t, half_rows, 2, "half rows");
ABSL_FLAG(int32_t, half_cols, 3, "half cols");
ABSL_FLAG(int32_t, match_level, 3, "match level");
ABSL_FLAG(int32_t, refine_size, 1, "refine size");
ABSL_FLAG(double, min_zncc, 0.8, "min zncc");
ABSL_FLAG(double, min_depth, 0.4, "min_depth");

namespace sv::dsol {

void Run() {
  TimerSummary tm{"dsol"};
  WindowTiler tiler;

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

  PlayData data(dataset, index, nkfs, nlevels, false);
  const auto& camera = data.camera;
  LOG(INFO) << camera.Repr();

  SelectCfg sel_cfg;
  sel_cfg.sel_level = absl::GetFlag(FLAGS_sel_level);
  sel_cfg.cell_size = absl::GetFlag(FLAGS_cell_size);
  sel_cfg.min_grad = absl::GetFlag(FLAGS_min_grad);
  sel_cfg.max_grad = absl::GetFlag(FLAGS_max_grad);
  PixelSelector selector{sel_cfg};
  LOG(INFO) << selector.Repr();

  StereoCfg match_cfg;
  match_cfg.half_rows = absl::GetFlag(FLAGS_half_rows);
  match_cfg.half_cols = absl::GetFlag(FLAGS_half_cols);
  match_cfg.match_level = absl::GetFlag(FLAGS_match_level);
  match_cfg.refine_size = absl::GetFlag(FLAGS_refine_size);
  match_cfg.min_depth = absl::GetFlag(FLAGS_min_depth);
  match_cfg.min_zncc = absl::GetFlag(FLAGS_min_zncc);
  StereoMatcher matcher{match_cfg};
  LOG(INFO) << matcher.Repr();

  const ColorMap cmap = MakeCmapPlasma();

  for (int i = 0; i < nkfs; ++i) {
    Keyframe kf;
    kf.SetFrame(data.frames.at(i));

    int n_pixels{};
    {
      auto t = tm.Scoped("SelectPixels");
      n_pixels = selector.Select(kf.grays_l(), tbb);
    }
    LOG(INFO) << "pixels: " << n_pixels;

    kf.Precompute(selector.pixels(), camera);

    int n_matched{};
    {
      auto t = tm.Scoped("MatchStereo");
      n_matched = matcher.Match(kf, camera, tbb);
    }
    LOG(INFO) << "n_matched: " << n_matched;

    int n_disps{};
    {
      auto t = tm.Scoped("InitDisps");
      n_disps = kf.InitFromDisp(matcher.disps(), camera);
    }
    LOG(INFO) << "n_disps: " << n_disps;

    kf.UpdateStatusInfo();
    LOG(INFO) << kf.status().Repr();

    if (vis) {
      cv::Mat disp_l;
      cv::cvtColor(kf.grays_l().front(), disp_l, cv::COLOR_GRAY2BGR);
      DrawSelectedPixels(disp_l, selector.pixels(), CV_RGB(0, 255, 255), 1);
      DrawDisparities(disp_l,
                      matcher.disps(),
                      selector.pixels(),
                      cmap,
                      camera.Depth2Disp(matcher.cfg().min_depth),
                      2);
      tiler.Tile(fmt::format("left_{}", i), disp_l);
      tiler.Tile(fmt::format("right_{}", i), kf.grays_r().front());

      cv::waitKey(1);
    }
  }

  LOG(INFO) << tm.ReportAll(true);

  if (vis) {
    cv::waitKey(-1);
  }

  // Compute depth metric
  //  DepthMetrics dm;
  //  const auto& disps = matcher.disps();
  //  for (int gr = 0; gr < disps.rows; ++gr) {
  //    for (int gc = 0; gc < disps.cols; ++gc) {
  //      const auto pred = disps.at<int16_t>(gr, gc);
  //      if (pred < 0) continue;
  //      const auto px = selector.pixels().at(gr, gc);
  //      CHECK(!IsPixBad(px));
  //      const auto gt = camera.Depth2Disp(depth.at<float>(px));
  //      dm.Update(gt, pred);
  //    }
  //  }
  //  const auto metrics = dm.Comptue();
  //  LOG(INFO) << fmt::format("{}", metrics);
}

}  // namespace sv::dsol

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  sv::dsol::Run();
}

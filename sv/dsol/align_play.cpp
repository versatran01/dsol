#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <fmt/ranges.h>

#include "sv/dsol/align.h"
#include "sv/dsol/extra.h"
#include "sv/dsol/select.h"
#include "sv/dsol/viz.h"
#include "sv/dsol/window.h"
#include "sv/util/cmap.h"
#include "sv/util/logging.h"
#include "sv/util/ocv.h"
#include "sv/util/summary.h"

ABSL_FLAG(bool, tbb, false, "use tbb");
ABSL_FLAG(bool, vis, true, "visualization");
ABSL_FLAG(int32_t, index, 0, "dataset index");
ABSL_FLAG(std::string,
          dir,
          // "/home/chao/Workspace/dataset/vkitti/Scene01/clone",
          "/home/chao/Workspace/dataset/tartan_air/office/Easy/P000",
          "dataset dir");

ABSL_FLAG(int32_t, cell_size, 16, "cell size");
ABSL_FLAG(int32_t, sel_level, 1, "select level");
ABSL_FLAG(int32_t, min_grad, 8, "minimum gradient");
ABSL_FLAG(int32_t, max_grad, 64, "maximum gradient");
ABSL_FLAG(double, min_ratio, 0.0, "minimum ratio");
ABSL_FLAG(double, max_ratio, 1.0, "maximum ratio");
ABSL_FLAG(bool, reselect, true, "reselect if ratio too low");

ABSL_FLAG(bool, affine, false, "optimize affine");
ABSL_FLAG(bool, stereo, false, "optimize stereo");
ABSL_FLAG(int32_t, c2, 2, "gradient weight");
ABSL_FLAG(int32_t, dof, 4, "student t dof");
ABSL_FLAG(int32_t, max_outliers, 1, "max outliers allowed");
ABSL_FLAG(double, grad_factor, 1.5, "grad factor");

ABSL_FLAG(int32_t, init_level, 4, "level to start optimization");
ABSL_FLAG(int32_t, max_iters, 10, "num iters each level");
ABSL_FLAG(double, max_xs, 0.1, "max xs to stop");

ABSL_FLAG(int32_t, num_kfs, 4, "num keyframes");
ABSL_FLAG(int32_t, num_levels, 5, "num pyramid levels");
ABSL_FLAG(int32_t, skip, 0, "num frames to skip");

ABSL_FLAG(double, max_depth, 100.0, "max depth to init");
ABSL_FLAG(double, min_depth, 2.0, "min depth to viz");
ABSL_FLAG(std::string, cm, "plasma", "colormap name");
ABSL_FLAG(bool, robust, false, "robust align");

ABSL_FLAG(double, pos, 0.2, "position offset added to current frame");

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

  PlayCfg play_cfg;
  play_cfg.index = absl::GetFlag(FLAGS_index);
  play_cfg.skip = absl::GetFlag(FLAGS_skip);
  const int nkfs = absl::GetFlag(FLAGS_num_kfs);
  play_cfg.nlevels = absl::GetFlag(FLAGS_num_levels);
  play_cfg.nframes = nkfs + 1;
  play_cfg.affine = absl::GetFlag(FLAGS_affine);
  LOG(INFO) << play_cfg.Repr();

  const double max_depth = absl::GetFlag(FLAGS_max_depth);
  const double pos_err = absl::GetFlag(FLAGS_pos);
  LOG(INFO) << "max_depth: " << max_depth;
  LOG(INFO) << "pos_err: " << pos_err;

  PlayData data(dataset, play_cfg);
  const auto& camera = data.camera;
  LOG(INFO) << camera.Repr();

  KeyframeWindow window(nkfs);

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

  // Make Aligner
  AlignCfg align_cfg;
  align_cfg.cost.affine = play_cfg.affine;
  align_cfg.cost.stereo = absl::GetFlag(FLAGS_stereo);
  align_cfg.cost.c2 = absl::GetFlag(FLAGS_c2);
  align_cfg.cost.dof = absl::GetFlag(FLAGS_dof);
  align_cfg.cost.max_outliers = absl::GetFlag(FLAGS_max_outliers);
  align_cfg.cost.grad_factor = absl::GetFlag(FLAGS_grad_factor);

  align_cfg.optm.max_iters = absl::GetFlag(FLAGS_max_iters);
  align_cfg.optm.init_level = absl::GetFlag(FLAGS_init_level);
  align_cfg.optm.max_xs = absl::GetFlag(FLAGS_max_xs);

  FrameAligner aligner{align_cfg};
  LOG(INFO) << aligner.Repr();

  // Init kf but not the last one
  for (int k = 0; k < nkfs; ++k) {
    LOG(INFO) << fmt::format("Keyframe: {}", k);
    auto& kf = window.AddKeyframe(data.frames.at(k));

    cv::Mat depth;
    if (max_depth > 0) {
      ThresholdDepth(data.depths.at(k), depth, max_depth);
    } else {
      depth = data.depths.at(k);
    }
    InitKfWithDepth(kf, camera, selector, depth, tm, tbb);

    LOG(INFO) << kf.status().Repr();
  }
  LOG(INFO) << window.Repr();

  Frame& frame = data.frames.back();

  LOG(INFO) << fmt::format(LogColor::kGreen, "[GT] {}", frame.state().Repr());

  LOG(INFO) << "Initialize from offset: " << pos_err;
  frame.state_.T_w_cl.translation().z() += pos_err;

  LOG(INFO) << fmt::format(
      LogColor::kYellow, "[INIT] {}", frame.state().Repr());

  const auto robust = absl::GetFlag(FLAGS_robust);
  LOG(INFO) << fmt::format("Robust: {}", robust);
  AlignStatus status;
  {
    auto t = tm.Scoped("Align");
    if (robust) {
      status = aligner.Align2(window.keyframes(), camera, frame, tbb);
    } else {
      status = aligner.Align(window.keyframes(), camera, frame, tbb);
    }
  }

  LOG(INFO) << fmt::format(LogColor::kBrightBlue, "{}", status.Repr());
  LOG(INFO) << fmt::format(
      LogColor::kBrightBlue, "tracks: {}", aligner.num_tracks());
  LOG(INFO) << fmt::format(
      LogColor::kBrightBlue, "[EST] {}", frame.state().Repr());

  // Print delta
  const Eigen::Vector3d dt =
      (data.poses.back().inverse() * frame.Twc()).translation();
  LOG(INFO) << fmt::format(LogColor::kBrightBlue,
                           "DELTA TRANS: {}, norm: {}",
                           dt.transpose(),
                           dt.norm());

  int n_mask = 0;
  {
    auto t = tm.Scoped("SetOccMask");
    n_mask = selector.SetOccMask(aligner.points1_vec());
  }
  LOG(INFO) << "curr frame n_mask: " << n_mask;

  Keyframe kf_new;
  kf_new.SetFrame(frame);
  const auto n_select = selector.Select(kf_new.grays_l(), tbb);
  kf_new.InitPoints(selector.pixels(), camera);
  LOG(INFO) << "Selection after masking: " << n_select;

  LOG(INFO) << fmt::format("align overlap: {}",
                           CountIdepths(aligner.idepths()));

  // Initialize this kf with align result
  int n_align = 0;
  {
    auto t = tm.Scoped("InitAlign");
    n_align = kf_new.InitFromAlign(aligner.idepths(), DepthPoint::kOkInfo - 1);
  }
  LOG(INFO) << "curr frame n_align: " << n_align;
  kf_new.InitPatches(tbb);

  for (int k = 0; k < window.size(); ++k) {
    LOG(INFO) << window.KfAt(k).status().Repr();
  }

  LOG(INFO) << tm.ReportAll(true);

  if (vis) {
    WindowTiler tiler{};
    const IntervalD range(0.0, 1.0 / absl::GetFlag(FLAGS_min_depth));
    const auto color = CV_RGB(0, 255, 255);
    const ColorMap cmap = GetColorMap(absl::GetFlag(FLAGS_cm));

    for (int k = 0; k < window.size(); ++k) {
      const auto& kf = window.KfAt(k);
      cv::Mat disp;
      cv::cvtColor(kf.gray_l(), disp, cv::COLOR_GRAY2BGR);
      DrawFramePoints(disp, kf.points(), cmap, range, 3);
      DrawSelectedPoints(disp, kf.points(), color, 1);

      tiler.Tile(fmt::format("keyframe{}", k), disp);
    }

    PyramidDisplay disp;
    disp.SetImages(frame.grays_l());
    for (int k = 0; k < window.size(); ++k) {
      const auto& points1 = aligner.points1_vec().at(k);
      DrawDepthPoints(disp.TopLevel(), points1, cmap, range, 2);
    }
    DrawSelectedPixels(disp.TopLevel(), selector.pixels(), color, 1);
    tiler.Tile("track", disp.canvas());

    // Draw the new kf
    cv::Mat disp_new;
    cv::cvtColor(frame.gray_l(), disp_new, cv::COLOR_GRAY2BGR);
    DrawFramePoints(disp_new, kf_new.points(), cmap, range, 3);
    DrawSelectedPixels(disp_new, selector.pixels(), color, 1);
    tiler.Tile("select", disp_new);
    tiler.Tile("mask", selector.mask());

    cv::waitKey(-1);
  }
}

}  // namespace sv::dsol

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  sv::dsol::Run();
}

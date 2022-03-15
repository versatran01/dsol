#include <absl/flags/flag.h>
#include <absl/flags/parse.h>

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
          "/home/chao/Workspace/dataset/tartan_air/carwelding/Easy/P001",
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
ABSL_FLAG(double, grad_factor, 1.0, "grad factor");

ABSL_FLAG(int32_t, max_iters, 5, "num iters each level");
ABSL_FLAG(int32_t, max_levels, 0, "num levels to optimize");
ABSL_FLAG(double, rel_change, 0.01, "rel_change");
ABSL_FLAG(double, pos, 0.4, "position offset added to current frame");

ABSL_FLAG(int32_t, num_kfs, 4, "num keyframes");
ABSL_FLAG(int32_t, num_levels, 5, "num pyramid levels");
ABSL_FLAG(double, max_depth, 100.0, "max depth to init");

namespace sv::dsol {

constexpr int kRepeat = 5;

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
  const int nkfs = absl::GetFlag(FLAGS_num_kfs);
  const int nlevels = absl::GetFlag(FLAGS_num_levels);
  const int nframes = nkfs + 1;
  const double max_depth = absl::GetFlag(FLAGS_max_depth);
  const double pos_err = absl::GetFlag(FLAGS_pos);
  const bool stereo = absl::GetFlag(FLAGS_stereo);
  const bool affine = absl::GetFlag(FLAGS_affine);

  LOG(INFO) << "index: " << index;
  LOG(INFO) << "num_kfs: " << nkfs;
  LOG(INFO) << "num_levels: " << nlevels;
  LOG(INFO) << "max_depth: " << max_depth;
  LOG(INFO) << "pos_err: " << pos_err;
  LOG(INFO) << "affine: " << affine;

  PlayData data(dataset, index, nframes, nlevels, affine);
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
  align_cfg.cost.affine = affine;
  align_cfg.cost.stereo = stereo;
  align_cfg.cost.c2 = absl::GetFlag(FLAGS_c2);
  align_cfg.cost.dof = absl::GetFlag(FLAGS_dof);
  align_cfg.cost.max_outliers = absl::GetFlag(FLAGS_max_outliers);
  align_cfg.cost.grad_factor = absl::GetFlag(FLAGS_grad_factor);

  align_cfg.solve.max_iters = absl::GetFlag(FLAGS_max_iters);
  align_cfg.solve.rel_change = absl::GetFlag(FLAGS_rel_change);

  FrameAligner aligner{align_cfg};
  LOG(INFO) << aligner.Repr();

  // Init kf but not the last one
  for (int k = 0; k < nkfs; ++k) {
    LOG(INFO) << fmt::format("Keyframe: {}, index: {}", k, index + k);
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

  AlignStatus status;
  {
    auto t = tm.Scoped("Align");
    status = aligner.Align(window.keyframes(), camera, frame, tbb);
  }

  LOG(INFO) << fmt::format(LogColor::kBrightBlue, "{}", status.Repr());
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
    auto t = tm.Scoped("CreateMask");
    n_mask = selector.CreateMask(aligner.points1_vec());
  }
  LOG(INFO) << "curr frame n_mask: " << n_mask;

  Keyframe kf_new;
  kf_new.SetFrame(frame);
  const auto n_select = selector.Select(kf_new.grays_l(), tbb);
  kf_new.Precompute(selector.pixels(), camera, tbb);
  LOG(INFO) << "Selection after masking: " << n_select;

  // Initialize this kf with align result
  int n_align = 0;
  {
    auto t = tm.Scoped("InitAlign");
    const auto idepth = aligner.CalcCellIdepth(selector.cfg().cell_size);
    n_align = kf_new.InitFromAlign(idepth, DepthPoint::kOkInfo);
  }
  LOG(INFO) << "curr frame n_align: " << n_align;

  for (int k = 0; k < window.size(); ++k) {
    LOG(INFO) << window.KfAt(k).status().Repr();
  }

  LOG(INFO) << tm.ReportAll(true);

  WindowTiler tiler{};

  if (vis) {
    const IntervalD range(0.0, 1.0 / 4.0);
    const auto color = CV_RGB(0, 255, 255);
    const ColorMap cmap = MakeCmapPlasma();

    for (int k = 0; k < window.size(); ++k) {
      const auto& kf = window.KfAt(k);
      cv::Mat disp;
      cv::cvtColor(kf.grays_l().front(), disp, cv::COLOR_GRAY2BGR);
      DrawFramePoints(disp, kf.points(), cmap, range, 2);
      tiler.Tile(fmt::format("kf{}", k), disp);
    }

    PyramidDisplay disp;
    disp.SetImages(frame.grays_l());
    for (const auto& warped : aligner.points1_vec()) {
      DrawDepthPoints(disp.TopLevel(), warped, cmap, range, 2);
    }
    DrawSelectedPixels(disp.TopLevel(), selector.pixels(), color, 1);
    tiler.Tile("frame_warp", disp.canvas());

    cv::Mat disp2;
    cv::cvtColor(frame.grays_l().front(), disp2, cv::COLOR_GRAY2BGR);
    DrawFramePoints(disp2, kf_new.points(), cmap, range, 2);
    DrawSelectedPixels(disp2, selector.pixels(), color, 1);
    tiler.Tile("frame_point", disp2);
    tiler.Tile("mask", selector.mask());

    cv::waitKey(-1);
  }
}

}  // namespace sv::dsol

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  sv::dsol::Run();
}

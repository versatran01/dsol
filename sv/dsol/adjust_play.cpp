#include <absl/flags/flag.h>
#include <absl/flags/parse.h>

#include "sv/dsol/adjust.h"
#include "sv/dsol/extra.h"
#include "sv/dsol/select.h"
#include "sv/dsol/viz.h"
#include "sv/dsol/window.h"
#include "sv/util/logging.h"
#include "sv/util/ocv.h"

ABSL_FLAG(bool, tbb, false, "use tbb");
ABSL_FLAG(bool, vis, true, "visualization");
ABSL_FLAG(int32_t, index, 0, "dataset index");
ABSL_FLAG(std::string,
          dir,
          "/home/chao/Workspace/dataset/vkitti/Scene01/clone",
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
ABSL_FLAG(int32_t, c2, 4, "gradient weight");
ABSL_FLAG(int32_t, dof, 4, "student t dof");
ABSL_FLAG(int32_t, max_outliers, 0, "max outliers allowed");
ABSL_FLAG(double, grad_factor, 1.0, "grad factor");

ABSL_FLAG(int32_t, max_iters, 5, "num iters each level");
ABSL_FLAG(int32_t, max_levels, 0, "num levels to optimize");
ABSL_FLAG(double, rel_change, 0.01, "rel_change");
ABSL_FLAG(double, pos, 0.0, "position offset added to keyframes");

ABSL_FLAG(bool, marg, false, "marginalization");
ABSL_FLAG(int32_t, num_kfs, 4, "num pyramid levels");
ABSL_FLAG(int32_t, num_levels, 3, "num pyramid levels");
ABSL_FLAG(int32_t, num_extra, 1, "num extra adjust runs");
ABSL_FLAG(double, max_depth, 100.0, "max depth to init");

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
  const int nkfs = absl::GetFlag(FLAGS_num_kfs);
  const int nlevels = absl::GetFlag(FLAGS_num_levels);
  const int nextra = absl::GetFlag(FLAGS_num_extra);
  const bool marg = absl::GetFlag(FLAGS_marg);
  const auto max_depth = absl::GetFlag(FLAGS_max_depth);
  const auto pos_err = absl::GetFlag(FLAGS_pos);
  const auto stereo = absl::GetFlag(FLAGS_stereo);
  const auto affine = absl::GetFlag(FLAGS_affine);

  CHECK_GE(index, 0);
  CHECK_GE(nkfs, 2);
  CHECK_GE(nextra, 0);
  CHECK_GT(nlevels, 0);

  LOG(INFO) << "index: " << index;
  LOG(INFO) << "num_kfs: " << nkfs;
  LOG(INFO) << "num_levels: " << nlevels;
  LOG(INFO) << "max_depth: " << max_depth;
  LOG(INFO) << "stereo: " << stereo;
  LOG(INFO) << "pos_err: " << pos_err;

  PlayData data(dataset, index, nkfs + nextra, nlevels, affine);
  const auto& camera = data.camera;
  LOG(INFO) << camera.Repr();

  KeyframeWindow window(nkfs);

  // Select
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

  // Adjust
  AdjustCfg adjust_cfg;
  adjust_cfg.cost.affine = absl::GetFlag(FLAGS_affine);
  adjust_cfg.cost.stereo = absl::GetFlag(FLAGS_stereo);
  adjust_cfg.cost.c2 = absl::GetFlag(FLAGS_c2);
  adjust_cfg.cost.dof = absl::GetFlag(FLAGS_dof);
  adjust_cfg.cost.max_outliers = absl::GetFlag(FLAGS_max_outliers);
  adjust_cfg.cost.grad_factor = absl::GetFlag(FLAGS_grad_factor);

  adjust_cfg.solve.max_iters = absl::GetFlag(FLAGS_max_iters);
  adjust_cfg.solve.max_levels = absl::GetFlag(FLAGS_max_levels);
  adjust_cfg.solve.rel_change = absl::GetFlag(FLAGS_rel_change);

  BundleAdjuster adjuster{adjust_cfg};
  LOG(INFO) << adjuster.Repr();

  int nba = 0;
  for (int i = 0; i < nkfs + nextra; ++i) {
    LOG(INFO) << fmt::format(LogColor::kGreen, "=== Adding Keyframe: {}", i);
    auto& kf = window.AddKeyframe(data.frames.at(i));

    cv::Mat depth;
    if (max_depth > 0) {
      ThresholdDepth(data.depths.at(i), depth, max_depth);
    } else {
      depth = data.depths.at(i);
    }

    InitKfWithDepth(kf, camera, selector, depth, tm, tbb);
    LOG(INFO) << kf.status().Repr();
    LOG(INFO) << window.Repr();

    if (adjuster.block().capacity() == 0) {
      const auto bytes = adjuster.Allocate(
          nkfs, static_cast<int>(selector.pixels().area() * 0.6));
      LOG(INFO) << fmt::format(LogColor::kRed,
                               "adjusert allocated {:.4f} MB",
                               static_cast<double>(bytes) * 1e-6);
      LOG(INFO) << adjuster.block().Repr();
    }

    if (window.full()) {
      // Bundle adjust
      LOG(INFO) << fmt::format(
          LogColor::kCyan, "=== Bundle Adjustment {}", nba);
      LOG(INFO) << "GT POS:  \n" << window.GetAllTrans().transpose();
      // Perturb last pose
      kf.state_.T_w_cl.translation().z() += pos_err;
      LOG(INFO) << "INIT POS:\n" << window.GetAllTrans().transpose();
      LOG(INFO) << "INIT AFF:\n" << window.GetAllAffine().transpose();

      AdjustStatus status;
      {  // Adjust
        auto t = tm.Scoped("Adjust");
        status = adjuster.Adjust(window.keyframes(), camera, tbb);
      }
      LOG(INFO) << "EST POS:\n" << window.GetAllTrans().transpose();
      LOG(INFO) << "EST AFF:\n" << window.GetAllAffine().transpose();

      for (int k = 0; k < window.size(); ++k) {
        const auto& kf1 = window.KfAt(k);
        const auto dt =
            (data.poses.at(k + nba).inverse() * kf1.Twc()).translation();
        LOG(INFO) << fmt::format(
            "kf {} dtrans: {}, \t norm: {}", k, dt.transpose(), dt.norm());
      }

      const int kf2rm = 0;
      if (marg) {
        LOG(INFO) << fmt::format(
            LogColor::kMagenta, "=== marginalizing kf {}", kf2rm);
        {
          auto t = tm.Scoped("Marginalize");
          adjuster.Marginalize(window.keyframes(), camera, kf2rm, tbb);
        }

        for (int k = 0; k < window.size(); ++k) {
          window.KfAt(k).SetFixed();
        }
      }

      // Remove keyframe
      window.RemoveKeyframeAt(kf2rm);
      LOG(INFO) << fmt::format(
          "Remove kf {}, window: {}", kf2rm, window.size());

      ++nba;
    }
  }

  LOG(INFO) << tm.ReportAll(true);

  //  if (vis) {
  //    const ColorMap cmap = MakeCmapPlasma();
  //    const IntervalD range(0.0, 1.0 / 4.0);

  //    for (int k = 0; k < window.size(); ++k) {
  //      const auto& kf = window.KfAt(k);
  //      cv::Mat disp;
  //      cv::cvtColor(kf.grays_l().front(), disp, cv::COLOR_GRAY2BGR);
  //      DrawFramePoints(disp, kf.points(), cmap, range, 2);
  //      Imshow("kf" + std::to_string(k), disp);
  //    }

  //    cv::waitKey(-1);
  //  }
}

}  // namespace sv::dsol

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  sv::dsol::Run();
}

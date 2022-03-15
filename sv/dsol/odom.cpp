#include "sv/dsol/odom.h"

#include <absl/strings/match.h>

#include <functional>
#include <opencv2/imgproc.hpp>

#include "sv/dsol/viz.h"
#include "sv/util/logging.h"
#include "sv/util/ocv.h"
#include "sv/util/summary.h"

namespace sv::dsol {

namespace {

// FIXEME (dsol): this is super hacky, maybe put them into a class
const ColorMap cmap = MakeCmapPlasma();
WindowTiler tiler{{1440, 900}, /*offset*/ {400, 600}, /*start*/ {400, 0}};
PyramidDisplay disp;
TimerSummary ts{"odom"};
StatsSummary ss{"odom"};

TimerSummary::StatsT SumStatsStartWith(const TimerSummary& tm,
                                       std::string_view start) {
  TimerSummary::StatsT stats;
  absl::Duration time;

  for (const auto& kv : tm.dict()) {
    if (absl::StartsWith(kv.first, start)) {
      time += kv.second.last();
    }
  }
  // should at least have something
  CHECK_NE(time, absl::ZeroDuration());
  stats.Add(time);
  return stats;
}

}  // namespace

void OdomCfg::Check() const {
  CHECK_GT(num_kfs, 1);
  CHECK_GT(num_levels, 1);
  CHECK_GT(min_track_ratio, 0);
  CHECK_LT(min_track_ratio, 1);
  CHECK_GT(vis_min_depth, 0);
  CHECK_EQ(init_depth || init_stereo, true);
}

std::string OdomCfg::Repr() const {
  return fmt::format(
      "OdomCfg(tbb={}, log={}, vis={}, marg={}, num_kfs={}, num_levels={}, "
      "min_track_ratio={}, min_track_per_kf={}, vis_min_depth={}, reinit={}, "
      "init_depth={}, init_stereo={}, init_align={})",
      tbb,
      log,
      vis,
      marg,
      num_kfs,
      num_levels,
      min_track_ratio,
      min_track_per_kf,
      vis_min_depth,
      reinit,
      init_depth,
      init_stereo,
      init_align);
}

/// ============================================================================
DirectOdometry::DirectOdometry(const OdomCfg& cfg) : cfg_{cfg} {
  cfg_.Check();
  window.Resize(cfg_.num_kfs);
}

void DirectOdometry::Init(const OdomCfg& cfg) {
  cfg_ = cfg;
  cfg_.Check();
  window.Resize(cfg_.num_kfs);
}

size_t DirectOdometry::Allocate(const ImagePyramid& grays, bool is_stereo) {
  VLOG(1) << "Allocating storage for everything";
  size_t bytes = 0;

  // frame (*2 if stereo)
  const auto frame_bytes =
      (static_cast<int>(is_stereo) + 1) * GetTotalBytes(grays);
  bytes += frame_bytes;
  VLOG(1) << "frame: " << frame_bytes;

  // selector
  const auto selector_bytes = selector.Allocate(grays);
  const auto grid_size = selector.cvsize();
  bytes += selector_bytes;
  VLOG(1) << "selector: " << selector_bytes;

  // matcher
  const auto matcher_bytes = matcher.Allocate(grid_size);
  bytes += matcher_bytes;
  VLOG(1) << "matcher: " << matcher_bytes;

  // aligner
  const auto aligner_bytes = aligner.Allocate(cfg_.num_kfs, grid_size);
  bytes += aligner_bytes;
  VLOG(1) << "aligner: " << aligner_bytes;

  // adjuster
  const auto adjuster_bytes = adjuster.Allocate(cfg_.num_kfs, grid_size.area());
  bytes += adjuster_bytes;
  VLOG(1) << "adjuster: " << adjuster_bytes;

  // window
  const auto window_bytes =
      window.Allocate(cfg_.num_kfs, cfg_.num_levels, grid_size) +
      cfg_.num_kfs * frame_bytes;
  bytes += window_bytes;
  VLOG(1) << "window: " << window_bytes;

  LOG(INFO) << fmt::format(LogColor::kBrightRed,
                           "Memory (MB): frame {:.4f}, select {:.4f}, match "
                           "{:.4f}, align {:.4f}, adjust {:.4f}, "
                           "window {:.4f}, total {:.4f}",
                           static_cast<double>(frame_bytes) / 1e6,
                           static_cast<double>(selector_bytes) / 1e6,
                           static_cast<double>(matcher_bytes) / 1e6,
                           static_cast<double>(aligner_bytes) / 1e6,
                           static_cast<double>(adjuster_bytes) / 1e6,
                           static_cast<double>(window_bytes) / 1e6,
                           static_cast<double>(bytes) / 1e6);

  return bytes;
}

std::string DirectOdometry::Repr() const {
  return fmt::format(
      "DirectOdometry(\n\t{}\n\t{}\n\t{}\n\t{}\n\t{}\n\t{}\n\t{})",
      cfg_.Repr(),
      camera.Repr(),
      selector.Repr(),
      matcher.Repr(),
      aligner.Repr(),
      adjuster.Repr(),
      window.Repr());
}

OdomStatus DirectOdometry::Estimate(const cv::Mat& image_l,
                                    const cv::Mat& image_r,
                                    const Sophus::SE3d& T_w_cl,
                                    const cv::Mat& depth) {
  OdomStatus status;

  status.track = Track(image_l, image_r, T_w_cl);
  if (cfg_.vis) DrawFrame(depth);

  if (!status.track.ok) Reinitialize();

  status.map = Map(status.track.add_kf, depth);
  if (cfg_.vis && status.track.add_kf) DrawKeyframe();

  if (cfg_.log > 0) Summarize(status.track.add_kf);
  return status;
}

TrackStatus DirectOdometry::Track(const cv::Mat& image_l,
                                  const cv::Mat& image_r,
                                  const Sophus::SE3d& T_w_cl) {
  TrackStatus status;
  // Make sure if we use affine in align, we also use it in adjust
  CHECK_EQ(aligner.cfg().cost.affine, adjuster.cfg().cost.affine);

  // Preprocess image (color -> gray -> vignette -> pyramid)
  CHECK(!image_l.empty());
  Preprocess(image_l, vign_l, grays_l);
  if (image_r.empty()) {
    CHECK(grays_r.empty());
  } else {
    Preprocess(image_r, vign_r, grays_r);
  }

  // Allocate storage
  if (total_bytes_ == 0) {
    total_bytes_ = Allocate(grays_l, !grays_r.empty());
  }

  // Update frame (keep the affine parameters)
  // Note that this uses the same storage as the static grays0 and grays1
  frame.SetGrays(grays_l, grays_r);
  frame.SetTwc(T_w_cl);

  // Get a copy of the current state if alignment failed
  const FrameState init_state = frame.state();

  // Track frame if possible
  status.ok = true;
  if (!window.empty()) {
    status.ok = TrackFrame();
  }

  // If tracking failed we revert state to the initial guess
  if (!status.ok) {
    LOG(INFO) << fmt::format(LogColor::kBrightRed,
                             "Tracking failed, restore initial state");
    frame.state() = init_state;
  }

  status.Twc = frame.Twc();
  // Determine whether we need to add a keyframe
  // If tracking failed we definitely need a new keyframe
  status.add_kf = !status.ok || ShouldAddKeyframe();
  return status;
}

MapStatus DirectOdometry::Map(bool add_kf, const cv::Mat& depth) {
  MapStatus status;

  // TODO (dsol): consider doing a 1 level adjustment when not adding new kf
  if (add_kf) {
    // No-op if we don't need a new keyframe
    // If window is full, we need to remove one first
    if (window.full()) {
      RemoveKeyframe();
      status.remove_kf = true;
    }

    // We then add a new kf to the window
    CHECK(!window.full());
    AddKeyframe(depth);
    ++total_kfs_;

    // Once added, we perform bundle adjustment with at least 2 kfs
    if (window.size() >= 2) {
      BundleAdjust();
    }
  }

  status.window_size = window.size();
  status.total_kfs = total_kfs_;
  return status;
}

void DirectOdometry::Preprocess(const cv::Mat& image,
                                const VignetteModel& vign,
                                ImagePyramid& grays) const {
  grays.resize(cfg_.num_levels);
  ConvertGray(image, grays[0]);
  vign.Correct(grays[0]);
  MakePyramid(grays);
}

void DirectOdometry::ConvertGray(const cv::Mat& image, cv::Mat& gray) const {
  CHECK(!image.empty());

  auto t = ts.Scoped("P0_ConvertGray");

  if (image.type() == CV_8UC3) {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  } else if (image.type() == CV_8UC1) {
    gray = image.clone();
  } else {
    CHECK(false) << "Invalid image type: " << CvTypeStr(image.type());
  }
}

void DirectOdometry::MakePyramid(ImagePyramid& grays) const {
  // Assumes we already put the original image into grays[0]
  CHECK(!grays.front().empty());
  CHECK_EQ(grays.front().type(), CV_8UC1);

  auto t = ts.Scoped("T0_MakePyramid");

  // Make pyramid
  for (int l = 1; l < cfg_.num_levels; ++l) {
    cv::pyrDown(grays[l - 1], grays[l]);
  }

  // go back and blur the first image to avoid double blurring it
  cv::GaussianBlur(grays[0], grays[0], {3, 3}, 0);
}

bool DirectOdometry::TrackFrame() {
  constexpr auto log_color = LogColor::kBrightMagenta;
  LOG(INFO) << fmt::format(
      log_color, "Track frame with window size: {}", window.size());

  AlignStatus status;
  {
    auto t = ts.Scoped("T1_TrackFrame");
    status = aligner.Align(window.keyframes(), camera, frame, cfg_.tbb);
  }

  LOG(INFO) << fmt::format(log_color, "{}", status.Repr());

  for (int k = 0; k < window.size(); ++k) {
    LOG(INFO) << k << ": " << window.KfAt(k).status().Repr();
  }

  // FIXME (dsol): This is a bit hacky
  // If affine is enabled, adjust is stereo but align is not, we need to also
  // update the right affine parameter, such that they don't deviate too much
  // from each other
  if (aligner.cfg().cost.affine && !aligner.cfg().cost.stereo &&
      adjuster.cfg().cost.stereo) {
    frame.state().affine_r = frame.state().affine_l;
  }

  // If num cost is too small then it is likely that tracking failed
  return status.num_costs >= (cfg_.num_kfs * cfg_.min_track_per_kf);
}

void DirectOdometry::Reinitialize() {
  constexpr auto log_color = LogColor::kCyan;

  if (!cfg_.reinit) {
    // Pause instead of quiting
    if (cfg_.vis) {
      LOG(INFO) << fmt::format(log_color,
                               "pause on reinitialization because reinit is "
                               "true, press any key to exit the program.");
      cv::waitKey(-1);
    }
    CHECK(false) << fmt::format(log_color, "Tracking failed, exit");
  }

  // Reset all kf, clear window, reset prior
  for (int k = 0; k < window.size(); ++k) {
    window.KfAt(k).Reset();
  }
  window.Clear();
  adjuster.ResetPrior();
  LOG(INFO) << fmt::format(log_color, "Reinitialize, clearing window");
}

bool DirectOdometry::ShouldAddKeyframe() const {
  // If window is not full, add kf anyway
  if (!window.full()) return true;

  // Get total number of tracked points from kf status
  int n_pixels = 0;
  int n_tracked = 0;
  for (int k = 0; k < window.size(); ++k) {
    const auto& kf_status = window.KfAt(k).status();
    n_tracked += kf_status.tracked;
    n_pixels += kf_status.pixels;
    // LOG(INFO) << fmt::format(
    // "kf {}, ratio: {}", k, kf_status.tracked / (kf_status.pixels + 1.0));
  }

  const auto ratio = static_cast<double>(n_tracked) / (n_pixels + 1.0);
  LOG(INFO) << fmt::format("num pixels: {}, num tracked {}, ratio: {:.2f}%",
                           n_pixels,
                           n_tracked,
                           ratio * 100);

  return ratio < cfg_.min_track_ratio;
}

void DirectOdometry::AddKeyframe(const cv::Mat& depth) {
  CHECK(!window.full());

  // Promote frame to keyframe
  auto& kf = window.AddKeyframe(frame);

  int n_mask{};
  {  // Create a mask from projections of map points
    auto t = ts.Scoped("K0_CreateMask");
    n_mask = selector.CreateMask(aligner.points1_vec());
  }

  int n_pixel{};
  {  // Select pixels with high gradients
    auto t = ts.Scoped("K1_SelectPixels");
    n_pixel = selector.Select(kf.grays_l(), cfg_.tbb);
  }

  {  // Precompute keyframe for direct image alignment
    auto t = ts.Scoped("K2_Precompute");
    kf.Precompute(selector.pixels(), camera, cfg_.tbb);
  }

  // Try to initialize depths of newly selected pixels
  // Initialize from depth image
  int n_init_depth{};
  if (cfg_.init_depth) {
    CHECK(!depth.empty());
    auto t = ts.Scoped("K3_InitDepths");
    n_init_depth = kf.InitFromDepth(depth, DepthPoint::kOkInfo);
  }

  // Initialize from stereo matching
  int n_init_stereo{};
  if (cfg_.init_stereo) {
    CHECK(frame.is_stereo()) << "Try to init from stereo but frame is mono";
    auto t = ts.Scoped("K4_InitStereo");
    n_init_stereo = matcher.Match(kf, camera, cfg_.tbb);
    kf.InitFromDisp(matcher.disps(), camera, DepthPoint::kOkInfo);
  }

  // Initialize points from Aligner results
  int n_init_align{};
  if (cfg_.init_align && !aligner.points1_vec().empty()) {
    auto t = ts.Scoped("K5_InitAlign");
    const auto idepth = aligner.CalcCellIdepth(selector.cfg().cell_size);
    n_init_align = kf.InitFromAlign(idepth, DepthPoint::kOkInfo - 1);
  }

  // Update status
  kf.UpdateStatusInfo();

  constexpr auto log_color = LogColor::kBrightBlue;
  LOG(INFO) << fmt::format(
      log_color,
      "PointInitStatus(maskes={}, select={}, depth={}, stereo={}, align={}, "
      "noinit={}, min_grad={})",
      n_mask,
      n_pixel,
      n_init_depth,
      n_init_stereo,
      n_init_align,
      n_pixel - n_init_depth - n_init_stereo - n_init_align,
      selector.cfg().min_grad);
  LOG(INFO) << fmt::format(log_color, "++ add keyframe");
  LOG(INFO) << fmt::format(log_color, "{}", kf.status().Repr());
}

void DirectOdometry::RemoveKeyframe() {
  CHECK_GE(window.size(), 2);
  constexpr auto log_color = LogColor::kBrightBlue;

  // Find the keyframe with the least tracked ratio, excluding latest keyframe
  int kf_ind = -1;
  double worst_ratio = 1.0;
  for (int k = 0; k < window.size() - 1; ++k) {
    // A simple heuristic that make newer keyframe has higher track ratio thus
    // more likely to remove older keyframe
    const auto ratio = window.KfAt(k).status().TrackRatio() * (1 + k * 0.05);
    if (ratio < worst_ratio) {
      kf_ind = k;
      worst_ratio = ratio;
    }
  }

  if (cfg_.marg) {
    LOG(INFO) << fmt::format(log_color, "-- Marginalizing kf {}", kf_ind);

    // Marginalize this frame and update prior
    {
      auto t = ts.Scoped("K6_Marginalize");
      adjuster.Marginalize(window.keyframes(), camera, kf_ind, cfg_.tbb);
    }

    // Fix first estimate for the rest of the keyframes
    // This will also fix the one to be marged but it will be reset on removal
    for (int k = 0; k < window.size(); ++k) {
      window.KfAt(k).SetFixed();
    }
  }

  // Finally emove keyframe from window
  const auto status = window.RemoveKeyframeAt(kf_ind);
  LOG(INFO) << fmt::format(log_color,
                           "-- Remove kf {} with lowest tracking ratio {:.2f}",
                           kf_ind,
                           worst_ratio * 100);
  LOG(INFO) << fmt::format(log_color, "-- {}", status.Repr());
}

void DirectOdometry::BundleAdjust() {
  constexpr auto log_color = LogColor::kBrightBlue;

  LOG(INFO) << fmt::format(
      log_color, "Adjust with window size: {}", window.size());

  adjuster.Allocate(window.max_kfs(), selector.pixels().area());
  AdjustStatus status;
  {
    auto t = ts.Scoped("K5_BundleAdjust");
    status = adjuster.Adjust(window.keyframes(), camera, cfg_.tbb);
  }
  LOG(INFO) << fmt::format(log_color, "{}", status.Repr());

  for (int k = 0; k < window.size(); ++k) {
    LOG(INFO) << k << ": " << window.KfAt(k).status().Repr();
  }

  // Set first kf's affine param to (0, 0) and adjust others accordingly. This
  // is to avoid unbounded growing of affine parameters
  const Eigen::Vector2d kf0_ab_l = window.KfAt(0).state().affine_l.ab;
  for (int k = 0; k < window.size(); ++k) {
    auto& state = window.KfAt(k).state();
    state.affine_l.ab -= kf0_ab_l;
    state.affine_r.ab -= kf0_ab_l;
  }

  LOG(INFO) << "\n" << window.GetAllAffine().transpose();

  // After bundle adjustment we need to update state of the current frame
  frame.SetState(window.CurrKf().state());
}

void DirectOdometry::Summarize(bool new_kf) {
  const auto stats_tracking = SumStatsStartWith(ts, "T");
  ts.Update("All_Tracking", stats_tracking);

  if (new_kf) {
    const auto stats_keyframe = SumStatsStartWith(ts, "K");
    ts.Update("All_Keyframe", stats_keyframe);
  }

  LOG_EVERY_N(INFO, cfg_.log) << ts.ReportAll(true);
}

void DirectOdometry::DrawFrame(const cv::Mat& depth) {
  tiler.Reset();

  static cv::Mat disp_frame;
  cv::cvtColor(frame.grays_l().front(), disp_frame, cv::COLOR_GRAY2BGR);
  const IntervalD range(0.0, 1.0 / cfg_.vis_min_depth);

  // Draw warped points on this frame, solid means successfully tracked
  for (const auto& warped : aligner.points1_vec()) {
    DrawDepthPoints(disp_frame, warped, cmap, range, 2);
  }
  tiler.Tile("frame_left", disp_frame);

  if (frame.is_stereo()) {
    tiler.Tile("frame_right", frame.grays_r().front());
  }

  if (!depth.empty()) {
    tiler.Tile(
        "idepth_left",
        ApplyCmap(cfg_.vis_min_depth / depth, 1.0, cv::COLORMAP_PINK, 0));
  }
}

void DirectOdometry::DrawKeyframe() const {
  static cv::Mat disp_kf;
  const auto& kf = window.CurrKf();
  cv::cvtColor(kf.grays_l().front(), disp_kf, cv::COLOR_GRAY2BGR);

  // Draw newly selected pixels and their initialized depths
  const auto cyan = CV_RGB(0, 255, 255);
  const IntervalD range(0.0, 1.0 / cfg_.vis_min_depth);
  DrawFramePoints(disp_kf, kf.points(), cmap, range, 3);
  DrawSelectedPixels(disp_kf, selector.pixels(), cyan, 1);

  tiler.Tile("kf", disp_kf);
  tiler.Tile("mask", selector.mask());

  // tmp, draw stereo matches
  cv::Mat disp_stereo;
  cv::cvtColor(kf.grays_l().front(), disp_stereo, cv::COLOR_GRAY2BGR);
  DrawDisparities(disp_stereo,
                  matcher.disps(),
                  selector.pixels(),
                  cmap,
                  camera.Depth2Disp(matcher.cfg().min_depth),
                  2);
  tiler.Tile("stereo", disp_stereo);

  cv::waitKey(1);
}

std::string OdomStatus::Repr() const {
  return fmt::format("OdomStatus(add_kf={}, remove_kf={}, total_kfs={})",
                     track.add_kf,
                     map.remove_kf,
                     map.total_kfs);
}

}  // namespace sv::dsol

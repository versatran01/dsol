#include "sv/dsol/frame.h"

#include "sv/dsol/pixel.h"
#include "sv/util/logging.h"
#include "sv/util/math.h"
#include "sv/util/tbb.h"

namespace sv::dsol {

Frame::Frame(const ImagePyramid& grays_l,
             const Sophus::SE3d& tf_w_cl,
             const AffineModel& affine_l)
    : grays_l_{grays_l}, state_{tf_w_cl, affine_l} {
  CHECK(IsImagePyramid(grays_l_));
}

std::string Frame::Repr() const {
  const auto size = cvsize();
  return fmt::format(
      "Frame(w={}, h={}, levels={}, stereo={}, trans=[{}], "
      "affine_l=[{}], affine_r=[{}])",
      size.width,
      size.height,
      levels(),
      is_stereo(),
      state_.T_w_cl.translation().transpose(),
      state_.affine_l.ab.transpose(),
      state_.affine_r.ab.transpose());
}

Frame::Frame(const ImagePyramid& grays_l,
             const ImagePyramid& grays_r,
             const Sophus::SE3d& tf_w_cl,
             const AffineModel& affine_l,
             const AffineModel& affine_r)
    : Frame(grays_l, tf_w_cl, affine_l) {
  grays_r_ = grays_r;
  state_.affine_r = affine_r;
  if (is_stereo()) {
    CHECK(IsStereoPair(grays_l_, grays_r_));
  }
}

void Frame::SetGrays(const ImagePyramid& grays_l, const ImagePyramid& grays_r) {
  grays_l_ = grays_l;
  grays_r_ = grays_r;

  CHECK(IsImagePyramid(grays_l_));
  if (is_stereo()) {
    CHECK(IsStereoPair(grays_l_, grays_r_));
  }
}

cv::Size Frame::cvsize() const noexcept {
  if (empty()) return {};
  const auto& img0 = grays_l_.at(0);
  return {img0.cols, img0.rows};
}

/// ============================================================================
size_t Keyframe::Allocate(int num_levels, const cv::Size& grid_size) {
  if (points_.empty()) {
    points_.resize(grid_size);
    patches_.resize(num_levels, PatchGrid{grid_size});
  } else {
    CHECK_EQ(points_.rows(), grid_size.height);
    CHECK_EQ(points_.cols(), grid_size.width);
  }

  return points_.size() * sizeof(FramePoint) +
         patches_.size() * patches_.front().size() * sizeof(Patch);
}

std::string Keyframe::Repr() const {
  const auto size = cvsize();
  return fmt::format(
      "Keyframe(w={}, h={}, levels={}, stereo={}, fixed={}, trans=[{}], "
      "affine_l=[{}], affine_r=[{}])",
      size.width,
      size.height,
      levels(),
      is_stereo(),
      is_fixed(),
      state_.T_w_cl.translation().transpose(),
      state_.affine_l.ab.transpose(),
      state_.affine_r.ab.transpose());
}

FrameState Keyframe::GetFirstEstimate() const noexcept {
  // We store the nominal state (eta = eta0 + x) together with the error state
  // (x). To get the first estimate eta0, we use eta - x. For frames without a
  // marginalization prior, the first estimate is the same as the current
  // linearization point.
  if (!fixed_) return state_;
  return state_ - x_;
}

FrameState Keyframe::GetEvaluationPoint() const noexcept {
  // Evaluation point is eta0 + x + delta. For frame without a marginalization
  // prior, the evaluation poitn is the same as the current state. However,
  // during bundle adjustment, for state with a marginalization prior, its delta
  // will be updated with the solution of the linear system, thus chaning the
  // evaluation point (of the cost function).
  if (!fixed_) return state_;
  return state_ + ErrorState{delta_};
}

void Keyframe::UpdateState(const Vector10dCRef& dx) noexcept {
  if (fixed_) {
    // For state with a marginalization prior, we just change delta (not
    // composing, because it will move the linearization point, thus it will be
    // different from the linearization point when marginalized).
    delta_ = dx;
  } else {
    // For state without prior, just update directly
    Frame::UpdateState(dx);
  }
}

void Keyframe::UpdatePoints(const VectorXdCRef& xm, int gsize) noexcept {
  //  for (int gr = 0; gr < points_.rows(); ++gr) {
  ParallelFor({0, points_.rows(), gsize}, [&](int gr) {
    for (int gc = 0; gc < points_.cols(); ++gc) {
      auto& point = points_.at(gr, gc);
      if (point.HidBad()) continue;

      const auto d_idepth = xm(point.hid);
      point.UpdateIdepth(d_idepth);
    }
  });
  //  }
}

void Keyframe::UpdateLinearizationPoint() noexcept {
  if (fixed_) {
    // First update linearization point with delta
    x_ += delta_;
    // Then update nominal state, this will not change first estimate
    Frame::UpdateState(delta_);
    // and reset delta
    delta_.setZero();
  } else {
    // If first estimate is not fixed, we don't need to do anything, just make
    // sure that delta is zero
    CHECK_EQ(delta_, Vector10d::Zero()) << delta_.transpose();
  }
}

void Keyframe::SetFrame(const Frame& frame) noexcept {
  // Reset status and fix
  Reset();
  SetState(frame.state());
  CopyImagePyramid(frame.grays_l(), grays_l_);
  if (frame.is_stereo()) {
    CopyImagePyramid(frame.grays_r(), grays_r_);
  }
}

void Keyframe::Precompute(const PixelGrid& pixels,
                          const Camera& camera,
                          int gsize) {
  CHECK(!empty());
  Allocate(levels(), pixels.cvsize());
  status_.pixels = InitPoints(pixels, camera);
  status_.patches = InitPatches(gsize);
}

int Keyframe::InitPoints(const PixelGrid& pixels, const Camera& camera) {
  CHECK_EQ(points_.rows(), pixels.rows());
  CHECK_EQ(points_.cols(), pixels.cols());

  // Reset all points to bad, including their depths
  points_.reset();

  int n_pixels = 0;
  for (int gr = 0; gr < points_.rows(); ++gr) {
    for (int gc = 0; gc < points_.cols(); ++gc) {
      const auto& px = pixels.at(gr, gc);

      // If a point is not selected, reset it
      if (IsPixBad(px)) continue;

      // Otherwise we just initialize it with the selected px. At its
      // current stage, it will not be used by either aligner or adjuster,
      // since idepth < 0. One needs to initialize idepth to something >= 0
      // and bump info to >= 0
      auto& point = points_.at(gr, gc);
      point.SetPix(px);
      point.SetNc(camera.Backward(point.uv()));
      ++n_pixels;
    }
  }
  return n_pixels;
}

int Keyframe::InitPatches(int gsize) {
  CHECK(!empty());
  CHECK(!points_.empty());
  CHECK(!patches_.empty());

  // Then prepare pyramid of patch grid
  const auto n_patches_total = ParallelReduce(
      {0, levels(), gsize},
      0,
      [&](int l, int& n_patches) {
        n_patches += InitPatchesLevel(l, gsize);
      },  // level
      std::plus<>{});

  return n_patches_total;
}

int Keyframe::InitPatchesLevel(int level, int gsize) {
  const auto& image = grays_l_.at(level);
  CHECK(!image.empty());

  auto& patches = patches_.at(level);
  CHECK_EQ(points_.rows(), patches.rows());
  CHECK_EQ(points_.cols(), patches.cols());

  const auto scale = PyrLevel2Scale(level);

  return ParallelReduce(
      {0, patches.rows(), gsize},
      0,
      [&](int gr, int& n_patches) {
        for (int gc = 0; gc < patches.cols(); ++gc) {
          const auto& point = points_.at(gr, gc);
          auto& patch = patches.at(gr, gc);

          // Check if we have a selected pixel at this point
          if (point.PixelBad()) {
            // Here we don't need to modify the patch, because point is
            // already bad, so we won't be using this patch in the first
            // place. But it's safer to invalidate this patch anyway.
            patch.SetBad();
            continue;
          }

          // Compute pixel at this pyramid level
          const auto px_s = ScalePix(point.px(), scale);

          // Pixel must be within image, due to the patch size we use a border
          // of 2. Points very close to image border are usually distorted both
          // geometrically and photometrically.
          if (IsPixOut(image, px_s, Patch::kBorder)) {
            // Due to scaling, it is possible that a good point will be too
            // close to border in a low res image, thus we cannot safely
            // extract a patch (without checking for bounds). For such cases
            // we will just invalidate this patch.
            patch.SetBad();
            continue;
          }

          // Finally we have a good pixel and we extract itensity and gradient
          patch.ExtractAround(image, px_s);
          ++n_patches;
        }  // gc
      },   // gr
      std::plus<>{});
}

int Keyframe::InitFromConst(double depth, double info) {
  CHECK_GT(depth, 0);
  CHECK(Precomputed());

  const auto idepth = 1.0 / depth;

  int n_init = 0;
  for (auto& point : points_) {
    // Skip bad or already initialized points
    if (point.SkipInit()) continue;
    point.SetIdepthInfo(idepth, info);
    ++n_init;
  }
  return n_init;
}

int Keyframe::InitFromDepth(const cv::Mat& depth, double info) {
  if (depth.empty()) return 0;

  CHECK(Precomputed());
  CHECK_EQ(depth.type(), CV_32FC1);
  CHECK_EQ(depth.rows, cvsize().height);
  CHECK_EQ(depth.cols, cvsize().width);

  int n_init = 0;
  for (int gr = 0; gr < points_.rows(); ++gr) {
    for (int gc = 0; gc < points_.cols(); ++gc) {
      auto& point = points_.at(gr, gc);
      // Skip bad or already initialized points
      if (point.SkipInit()) continue;

      const auto d = depth.at<float>(RoundPix(point.px()));

      if (d > 0.1) {  // Reject depth too close
        point.SetIdepthInfo(1.0 / d, info);
        ++n_init;
      }
    }  // gc
  }    // gr

  return n_init;
}

int Keyframe::InitFromDisp(const cv::Mat& disp,
                           const Camera& camera,
                           double info) {
  if (disp.empty()) return 0;

  CHECK(Precomputed());
  CHECK_EQ(disp.type(), CV_16SC1);
  CHECK_EQ(disp.rows, points_.rows());
  CHECK_EQ(disp.cols, points_.cols());

  int n_init = 0;
  for (int gr = 0; gr < points_.rows(); ++gr) {
    for (int gc = 0; gc < points_.cols(); ++gc) {
      auto& point = points_.at(gr, gc);
      if (point.SkipInit()) continue;

      // Skip invalid disparity
      const auto d = disp.at<int16_t>(gr, gc);
      if (d < 0) continue;

      const auto idepth = camera.Disp2Idepth(static_cast<double>(d));
      point.SetIdepthInfo(idepth, info);
      ++n_init;
    }  // gc
  }    // gr

  return n_init;
}

int Keyframe::InitFromAlign(const cv::Mat& idepth, double info) {
  if (idepth.empty()) return 0;

  CHECK(Precomputed());
  CHECK_EQ(idepth.type(), CV_64FC2);
  CHECK_EQ(idepth.rows, points_.rows());
  CHECK_EQ(idepth.cols, points_.cols());

  int n_init = 0;

  for (int gr = 0; gr < points_.rows(); ++gr) {
    for (int gc = 0; gc < points_.cols(); ++gc) {
      auto& point = points_.at(gr, gc);
      if (point.SkipInit()) continue;

      // Skip empty cell from input
      const auto& cell = idepth.at<cv::Vec2d>(gr, gc);
      if (cell[1] == 0) continue;

      point.SetIdepthInfo(cell[0] / cell[1], info);
      ++n_init;
    }  // gc
  }    // gr

  return n_init;
}

void Keyframe::Reset() noexcept {
  status_ = {};
  fixed_ = false;
  x_ = {};
  delta_.setZero();
}

/// ============================================================================
std::string KeyframeStatus::FrameStatus() {
  return fmt::format("FrameStatus(pixels={}, patches={})", pixels, patches);
}

std::string KeyframeStatus::TrackStatus() {
  return fmt::format("TrackStatus(outside={}, outlier={}, tracker={})",
                     outside,
                     outlier,
                     tracked);
}

std::string KeyframeStatus::PointStatus() {
  return fmt::format(
      "PointStatus(info_bad={}, info_uncert={}, info_ok={}, info_max={})",
      info_bad,
      info_uncert,
      info_ok,
      info_max);
}

std::string KeyframeStatus::Repr() const {
  return fmt::format(
      "KeyframeStatus(pixels={}, patches={} | "
      "outside={}, outlier={}, tracked={} | "
      "info_bad={}, info_uncert={}, info_ok={}, info_max={})",
      pixels,
      patches,
      outside,
      outlier,
      tracked,
      info_bad,
      info_uncert,
      info_ok,
      info_max);
}

void KeyframeStatus::UpdateInfo(const FramePointGrid& points0) {
  // Reset adjust related status
  info_bad = info_uncert = info_ok = info_max = 0;

  for (const auto& point : points0) {
    if (point.PixelBad()) continue;

    if (point.InfoMax()) {
      ++info_max;
    } else if (point.InfoOk()) {
      ++info_ok;
    } else if (!point.InfoBad()) {
      ++info_uncert;
    } else {
      ++info_bad;
    }
  }
  CHECK_EQ(pixels, info_max + info_ok + info_uncert + info_bad) << Repr();
}

void KeyframeStatus::UpdateTrack(const DepthPointGrid& points1) {
  // Reset align related status
  outside = outlier = tracked = 0;

  for (const auto& point : points1) {
    if (point.PixelBad()) continue;
    if (point.InfoOk()) {
      ++tracked;
    } else if (point.info() == DepthPoint::kMinInfo) {
      ++outside;
    } else {
      ++outlier;
    }
  }
  CHECK_EQ(info_ok + info_max, tracked + outside + outlier) << Repr();
}

Keyframe& GetKfAt(KeyframePtrSpan keyframes, int k) {
  auto* pkf = keyframes.at(k);
  CHECK_NOTNULL(pkf);
  auto& kf = *pkf;
  CHECK(kf.Ok());
  return kf;
}

const Keyframe& GetKfAt(KeyframePtrConstSpan keyframes, int k) {
  const auto* pkf = keyframes.at(k);
  CHECK_NOTNULL(pkf);
  const auto& kf = *pkf;
  CHECK(kf.Ok());
  return kf;
}

std::string FrameState::Repr() const {
  return fmt::format("State(quat=[{}], trans=[{}], aff_l=[{}], aff_r=[{}])",
                     T_w_cl.unit_quaternion().coeffs().transpose(),
                     T_w_cl.translation().transpose(),
                     affine_l.ab.transpose(),
                     affine_r.ab.transpose());
}

}  // namespace sv::dsol

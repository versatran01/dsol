#include "sv/dsol/frame.h"

#include "sv/dsol/pixel.h"
#include "sv/util/logging.h"
#include "sv/util/math.h"
#include "sv/util/tbb.h"

namespace sv::dsol {

std::string Frame::Repr() const {
  const auto size = image_size();
  return fmt::format(
      "Frame(w={}, h={}, levels={}, stereo={}, trans=[{}], "
      "affine_l=[{}], affine_r=[{}])",
      size.width,
      size.height,
      levels(),
      is_stereo(),
      fmt::streamed(state_.T_w_cl.translation().transpose()),
      fmt::streamed(state_.affine_l.ab.transpose()),
      fmt::streamed(state_.affine_r.ab.transpose()));
}

Frame::Frame(const ImagePyramid& grays_l,
             const Sophus::SE3d& tf_w_cl,
             const AffineModel& affine_l)
    : grays_l_{grays_l}, state_{tf_w_cl, affine_l, {0, 0}} {
  CHECK(IsImagePyramid(grays_l_));
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

cv::Size Frame::image_size() const noexcept {
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
  const auto size = image_size();
  return fmt::format(
      "Keyframe(w={}, h={}, levels={}, stereo={}, fixed={}, trans=[{}], "
      "affine_l=[{}], affine_r=[{}])",
      size.width,
      size.height,
      levels(),
      is_stereo(),
      is_fixed(),
      fmt::streamed(state_.T_w_cl.translation().transpose()),
      fmt::streamed(state_.affine_l.ab.transpose()),
      fmt::streamed(state_.affine_r.ab.transpose()));
}

FrameState Keyframe::GetFirstEstimate() const noexcept {
  // We store the nominal state (eta = eta0 + x) together with the error state
  // (x). To get the first estimate eta0, we use eta - x. For frames without a
  // marginalization prior, the first estimate is the same as the current
  // linearization point.
  if (!fixed_) return state_;
  return state_ - x_;
}

void Keyframe::UpdateState(const Vector10dCRef& dx) noexcept {
  // For state with a marginalization prior, we only update x
  if (fixed_) x_ += dx;
  Frame::UpdateState(dx);
}

void Keyframe::UpdatePoints(const VectorXdCRef& xm, double scale, int gsize) {
  ParallelFor({0, points_.rows(), gsize}, [&](int gr) {
    for (int gc = 0; gc < points_.cols(); ++gc) {
      auto& point = points_.at(gr, gc);
      if (point.HidBad()) continue;
      point.UpdateIdepth(xm[point.hid] * scale);
    }
  });
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

int Keyframe::InitPoints(const PixelGrid& pixels, const Camera& camera) {
  Allocate(levels(), pixels.cvsize());

  // Reset all points to bad, including their depths
  points_.reset();

  int n_pixels = 0;
  for (int gr = 0; gr < points_.rows(); ++gr) {
    for (int gc = 0; gc < points_.cols(); ++gc) {
      const auto& px = pixels.at(gr, gc);

      // If a point is not selected, reset it
      if (IsPixOut(image_size(), px, 1)) continue;

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

  status_.pixels = n_pixels;
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

  status_.patches = n_patches_total;
  return n_patches_total;
}

int Keyframe::InitPatchesLevel(int level, int gsize) {
  const auto& image = grays_l_.at(level);
  CHECK(!image.empty());

  auto& patches = patches_.at(level);
  CHECK_EQ(points_.rows(), patches.rows());
  CHECK_EQ(points_.cols(), patches.cols());

  if (level == 0) {
    return ParallelReduce(
        {0, patches.rows(), gsize},
        0,
        [&](int gr, int& n_patches) {
          for (int gc = 0; gc < patches.cols(); ++gc) {
            const auto& point = points_.at(gr, gc);
            auto& patch = patches.at(gr, gc);
            patch.SetBad();

            // Check if we have a selected pixel at this point
            if (point.PixelBad()) continue;

            // Make sure we are inside image with border of 1, because
            // gradient is computed using Scharr operator which is 3x3. We can
            // safely apply it without checking for bounds.
            CHECK(IsPixIn(image, point.px(), 1));

            patch.ExtractAround3(image, point.px());
            ++n_patches;
          }  // gc
        },   // gr
        std::plus<>{});
  }

  const auto scale = PyrLevel2Scale(level);
  return ParallelReduce(
      {0, patches.rows(), gsize},
      0,
      [&](int gr, int& n_patches) {
        for (int gc = 0; gc < patches.cols(); ++gc) {
          const auto& point = points_.at(gr, gc);
          auto& patch = patches.at(gr, gc);
          patch.SetBad();

          // Check if we have a selected pixel at this point
          if (point.PixelBad()) {
            // Here we don't need to modify the patch, because point is
            // already bad, so we won't be using this patch in the first
            // place. But it's safer to invalidate this patch anyway.
            continue;
          }

          // Compute pixel at this pyramid level
          const auto px_s = ScalePix(point.px(), scale);

          // Pixel must be within image, due to the patch size we use a border
          // of 2 (1 for border pixel and 1 for gradient). Points very close to
          // image border are usually distorted both geometrically and
          // photometrically.
          if (IsPixOut(image, px_s, 2)) {
            // Due to scaling, it is possible that a good point will be too
            // close to border in a low res image, thus we cannot safely
            // extract a patch (without checking for bounds). For such cases
            // we will just invalidate this patch.
            continue;
          }

          // Finally we have a good pixel and we extract intensity and gradient
          patch.ExtractAround(image, px_s);
          ++n_patches;
        }  // gc
      },   // gr
      std::plus<>{});
}

int Keyframe::InitFromConst(double depth, double info) {
  CHECK_GT(depth, 0);
  CHECK(Ok());

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

  CHECK(Ok());
  CHECK_EQ(depth.type(), CV_32FC1);
  CHECK_EQ(depth.rows, image_size().height);
  CHECK_EQ(depth.cols, image_size().width);

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

  status_.depths += n_init;
  return n_init;
}

int Keyframe::InitFromDisp(const cv::Mat& disp,
                           const Camera& camera,
                           double info) {
  if (disp.empty()) return 0;

  CHECK(Ok());
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

  status_.depths += n_init;
  return n_init;
}

int Keyframe::InitFromAlign(const cv::Mat& idepth, double info) {
  if (idepth.empty()) return 0;

  CHECK(Ok());
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
      if (cell[1] <= 0) continue;

      point.SetIdepthInfo(cell[0] / cell[1], info);
      ++n_init;
    }  // gc
  }    // gr

  status_.depths += n_init;
  return n_init;
}

void Keyframe::Reset() noexcept {
  status_ = {};
  fixed_ = false;
  x_ = {};
}

/// ============================================================================
std::string KeyframeStatus::FrameStatus() const {
  return fmt::format(
      "pixels={:4d}, depths={:4d}, patches={:4d}", pixels, depths, patches);
}

std::string KeyframeStatus::PointStatus() const {
  return fmt::format(
      "info_bad={:3d}, info_uncert={:3d}, info_ok={:3d}, info_max={:3d}",
      info_bad,
      info_uncert,
      info_ok,
      info_max);
}

std::string KeyframeStatus::Repr() const {
  return fmt::format("KeyframeStatus({} | {})", FrameStatus(), PointStatus());
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
  // Make sure all points add up to the number of pixels
  CHECK_EQ(pixels, info_max + info_ok + info_uncert + info_bad) << Repr();
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
  return fmt::format(
      "State(quat=[{}], trans=[{}], aff_l=[{}], aff_r=[{}])",
      fmt::streamed(T_w_cl.unit_quaternion().coeffs().transpose()),
      fmt::streamed(T_w_cl.translation().transpose()),
      fmt::streamed(affine_l.ab.transpose()),
      fmt::streamed(affine_r.ab.transpose()));
}

cv::Rect2d GetMinBboxInfoGe(const FramePointGrid& points, double min_info) {
  static constexpr auto kF64Max = std::numeric_limits<double>::max();

  double min_x = kF64Max;
  for (int gc = 0; gc < points.cols(); ++gc) {
    if (min_x < kF64Max) break;
    for (int gr = 0; gr < points.rows(); ++gr) {
      const auto& point = points.at(gr, gc);
      if (point.info() < min_info) continue;
      CHECK(point.PixelOk());
      CHECK(point.DepthOk());
      // Now we try to find the point with smallest y
      if (point.px().x < min_x) min_x = point.px().x;
    }
  }

  double min_y = kF64Max;
  for (int gr = 0; gr < points.rows(); ++gr) {
    if (min_y < kF64Max) break;
    for (int gc = 0; gc < points.cols(); ++gc) {
      const auto& point = points.at(gr, gc);
      if (point.info() < min_info) continue;
      CHECK(point.PixelOk());
      CHECK(point.DepthOk());
      if (point.px().y < min_y) min_y = point.px().y;
    }
  }

  double max_x = 0;
  for (int gc = points.cols() - 1; gc >= 0; --gc) {
    if (max_x > 0) break;
    for (int gr = 0; gr < points.rows(); ++gr) {
      const auto& point = points.at(gr, gc);
      if (point.info() < min_info) continue;
      CHECK(point.PixelOk());
      CHECK(point.DepthOk());
      if (point.px().x > max_x) max_x = point.px().x;
    }
  }

  double max_y = 0;
  for (int gr = points.rows() - 1; gr >= 0; --gr) {
    if (max_y > 0) break;
    for (int gc = 0; gc < points.cols(); ++gc) {
      const auto& point = points.at(gr, gc);
      if (point.info() < min_info) continue;
      CHECK(point.PixelOk());
      CHECK(point.DepthOk());
      if (point.px().y > max_y) max_y = point.px().y;
    }
  }

  return {min_x, min_y, max_x - min_x, max_y - min_y};
}

}  // namespace sv::dsol

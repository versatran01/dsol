#pragma once

#include <absl/types/span.h>

#include <sophus/se3.hpp>

#include "sv/dsol/camera.h"
#include "sv/dsol/dim.h"
#include "sv/dsol/image.h"
#include "sv/dsol/point.h"

namespace sv::dsol {

struct AffineModel {
  AffineModel(double a_in = 0, double b_in = 0) : ab{a_in, b_in} {}
  Eigen::Vector2d ab{Eigen::Vector2d::Zero()};
  double a() const { return ab[0]; }
  double b() const { return ab[1]; }
};

/// @brief Frame error state map
struct ErrorState {
  using Vector10d = Eigen::Matrix<double, Dim::kFrame, 1>;
  using Vector10dCRef = Eigen::Ref<const Vector10d>;

  Vector10d x_{Vector10d::Zero()};

  ErrorState() = default;
  explicit ErrorState(const Vector10dCRef& delta) : x_{delta} {}
  const Vector10d& vec() const noexcept { return x_; }
  Eigen::Vector2d ab_l() const noexcept { return x_.segment<2>(Dim::kPose); }
  Eigen::Vector2d ab_r() const noexcept { return x_.segment<2>(Dim::kMono); }
  Sophus::SE3d dT() const noexcept {
    return {Sophus::SO3d::exp(x_.head<3>()), x_.segment<3>(3)};
  }

  ErrorState& operator+=(const Vector10d& dx) noexcept {
    x_ += dx;
    return *this;
  }
};

/// @brief Frame state
struct FrameState {
  Sophus::SE3d T_w_cl;   // pose of left camera
  AffineModel affine_l;  // affine of left image
  AffineModel affine_r;  // affine of right image

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const FrameState& rhs) {
    return os << rhs.Repr();
  }

  FrameState& operator+=(const ErrorState& er) noexcept {
    T_w_cl *= er.dT();
    affine_l.ab += er.ab_l();
    affine_r.ab += er.ab_r();
    return *this;
  }

  FrameState& operator-=(const ErrorState& er) noexcept {
    T_w_cl *= er.dT().inverse();
    affine_l.ab -= er.ab_l();
    affine_r.ab -= er.ab_r();
    return *this;
  }

  friend FrameState operator+(FrameState st, const ErrorState& er) noexcept {
    return st += er;
  }

  friend FrameState operator-(FrameState st, const ErrorState& er) noexcept {
    return st -= er;
  }
};

/// @brief a simple frame
struct Frame {
  using Vector10d = ErrorState::Vector10d;
  using Vector10dCRef = ErrorState::Vector10dCRef;

  // images
  ImagePyramid grays_l_;
  ImagePyramid grays_r_;
  FrameState state_;

  Frame() = default;
  virtual ~Frame() noexcept = default;

  /// @brief Mono Ctor
  Frame(const ImagePyramid& grays_l,
        const Sophus::SE3d& tf_w_cl,
        const AffineModel& affine_l = {});

  /// @brief Stereo Ctor
  Frame(const ImagePyramid& grays_l,
        const ImagePyramid& grays_r,
        const Sophus::SE3d& tf_w_cl,
        const AffineModel& affine_l = {},
        const AffineModel& affine_r = {});

  virtual std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const Frame& rhs) {
    return os << rhs.Repr();
  }

  /// @brief Info
  int levels() const noexcept { return static_cast<int>(grays_l_.size()); }
  bool empty() const noexcept { return grays_l_.empty(); }
  bool is_stereo() const noexcept { return !grays_r_.empty(); }
  cv::Size cvsize() const noexcept;

  /// @brief Accessors
  const ImagePyramid& grays_l() const noexcept { return grays_l_; }
  const ImagePyramid& grays_r() const noexcept { return grays_r_; }
  const cv::Mat& gray_l() const noexcept { return grays_l_.front(); }

  FrameState& state() noexcept { return state_; }
  const FrameState& state() const noexcept { return state_; }
  const Sophus::SE3d& Twc() const noexcept { return state_.T_w_cl; }

  /// @brief Modifiers
  void SetGrays(const ImagePyramid& grays_l, const ImagePyramid& grays_r = {});
  void SetTwc(const Sophus::SE3d& T_w_cl) noexcept { state_.T_w_cl = T_w_cl; }
  void SetState(const FrameState& state) noexcept { state_ = state; }
  virtual void UpdateState(const Vector10dCRef& dx) noexcept {
    state_ += ErrorState{dx};
  }
};

/// @brief Stores variaus info of this keyframe
struct KeyframeStatus {
  // frame
  int pixels{};   // num selected pixels
  int depths{};   // num init depths
  int patches{};  // num valid patches (in all pyr)
  // point0 info
  int info_bad{};     // num bad points
  int info_uncert{};  // num uncertain points [0, ok)
  int info_ok{};      // num good points [ok, max)
  int info_max{};     // num max points == max

  std::string FrameStatus() const;
  std::string TrackStatus() const;
  std::string PointStatus() const;

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const KeyframeStatus& rhs) {
    return os << rhs.Repr();
  }

  /// @brief Update depth info status
  void UpdateInfo(const FramePointGrid& points0);
};

/// @brief a keyframe is a frame with depth at features
struct Keyframe final : public Frame {
  KeyframeStatus status_{};
  FramePointGrid points_{};
  std::vector<PatchGrid> patches_{};  // precomputed patches
  bool fixed_{false};                 // whether first estimate is fixed
  ErrorState x_{};                    // error state x in dso paper

  /// @brief Fix first estimate
  bool is_fixed() const noexcept { return fixed_; }
  void SetFixed() noexcept { fixed_ = true; }
  /// @brief This is eta0 in dso paper
  FrameState GetFirstEstimate() const noexcept;

  /// @brief Update state during optimization, need to call
  /// UpdateLinearizationPoint() to finalize the change
  void UpdateState(const Vector10dCRef& dx) noexcept override;
  void UpdatePoints(const VectorXdCRef& xm, int gsize = 0) noexcept;
  void UpdateStatusInfo() noexcept { status_.UpdateInfo(points_); }

  FramePointGrid& points() noexcept { return points_; }
  const KeyframeStatus& status() const noexcept { return status_; }
  const FramePointGrid& points() const noexcept { return points_; }
  const std::vector<PatchGrid>& patches() const noexcept { return patches_; }

  Keyframe() = default;
  std::string Repr() const override;

  /// @brief Initialize from frame
  void SetFrame(const Frame& frame) noexcept;

  /// @brief Allocate storage for points and patches, not for images
  /// @return number of bytes
  size_t Allocate(int num_levels, const cv::Size& grid_size);
  /// @brief Initialize points (pixels only)
  int InitPoints(const PixelGrid& pixels, const Camera& camera);

  /// @group Initialize point depth from various sources
  int InitFromConst(double depth, double info = DepthPoint::kOkInfo);
  /// @brief Initialize point depth from depths (from RGBD or ground truth)
  int InitFromDepth(const cv::Mat& depth, double info = DepthPoint::kOkInfo);
  /// @brief Initialize point depth from disparities (from StereoMatcher)
  int InitFromDisp(const cv::Mat& disp,
                   const Camera& camera,
                   double info = DepthPoint::kOkInfo);
  /// @brief Initialize point depth from inverse depths (from FrameAligner)
  int InitFromAlign(const cv::Mat& idepth, double info);

  /// @brief Initialize patches
  /// @return number of patches from all levels
  int InitPatches(int gsize = 0);
  /// @brief Initialize patches at level
  /// @return number of precomputed patches within this level
  int InitPatchesLevel(int level, int gsize = 0);

  /// @brief Reset this keyframe
  void Reset() noexcept;
  bool Ok() const noexcept { return status_.pixels > 0; }
};

using KeyframePtrSpan = absl::Span<Keyframe*>;
using KeyframePtrConstSpan = absl::Span<Keyframe const* const>;

/// @brief Get a reference to the k-th keyframe, with not-null and ok checks
Keyframe& GetKfAt(KeyframePtrSpan keyframes, int k);
const Keyframe& GetKfAt(KeyframePtrConstSpan keyframes, int k);

/// @brief Get the smallest bounding box that covers all points with
/// info >= min_info
cv::Rect2d GetMinBboxInfoGe(const FramePointGrid& points, double min_info);

}  // namespace sv::dsol

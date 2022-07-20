#pragma once

#include "sv/dsol/adjust.h"
#include "sv/dsol/align.h"
#include "sv/dsol/select.h"
#include "sv/dsol/stereo.h"
#include "sv/dsol/window.h"
#include "sv/util/cmap.h"

namespace sv::dsol {

struct OdomCfg {
  int tbb{0};                   // tbb grainsize
  int log{0};                   // log interval
  int vis{0};                   // show visualization
  int num_kfs{4};               // num kfs in window
  int num_levels{4};            // num pyramid levels
  double min_track_ratio{0.3};  // min track ratio to add new kf
  double vis_min_depth{4.0};    // min depth in visualization
  bool marg{false};             // enable marginalization
  bool reinit{false};           // reinitialize upon tracking failure
  bool init_depth{true};        // init from depth
  bool init_stereo{false};      // init from stereo
  bool init_align{false};       // init from align

  void Check() const;
  std::string Repr() const;
};

struct TrackStatus {
  Sophus::SE3d Twc{};  // current estimate
  bool add_kf{false};  // need to add a new kf
  bool ok{false};      // tracking ok
};

struct MapStatus {
  bool remove_kf{false};  // whether we removed a kf from window
  int total_kfs{};        // total number of kfs added
  int window_size{};      // current window size
};

struct OdomStatus {
  TrackStatus track;
  MapStatus map;

  const auto& Twc() const noexcept { return track.Twc; }
  std::string Repr() const;
};

/// @brief Simple dsol odometry
struct DirectOdometry {
 private:
  OdomCfg cfg_;           // config
  int total_kfs_{};       // total keyframes
  size_t total_bytes_{};  // number of bytes used

 public:
  Frame frame;
  Camera camera;
  KeyframeWindow window;
  PixelSelector selector;
  StereoMatcher matcher;
  FrameAligner aligner;
  BundleAdjuster adjuster;

  ImagePyramid grays_l;
  ImagePyramid grays_r;
  VignetteModel vign_l;
  VignetteModel vign_r;

  ColorMap cmap;

  /// @brief Ctor
  explicit DirectOdometry(const OdomCfg& cfg = {});
  /// @brief Initialize window
  void Init(const OdomCfg& cfg);
  /// @brief Allocate storage
  size_t Allocate(const ImagePyramid& grays, bool is_stereo);

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const DirectOdometry& rhs) {
    return os << rhs.Repr();
  }

  void SetCamera(const Camera& cam) noexcept { camera = cam; }

  /// @brief Estimate odometry full version
  OdomStatus Estimate(const cv::Mat& image_l,
                      const cv::Mat& image_r,
                      const Sophus::SE3d& dT,
                      const cv::Mat& depth = {});
  TrackStatus Track(const cv::Mat& image_l,
                    const cv::Mat& image_r,
                    const Sophus::SE3d& dT);
  MapStatus Map(bool add_kf, const cv::Mat& depth);

 private:
  /// @brief Preprocess images, convert to gray and create pyramid
  void Preprocess(const cv::Mat& image,
                  const VignetteModel& vign,
                  ImagePyramid& grays) const;
  /// @brief Convert to gray if image is color
  void ConvertGray(const cv::Mat& image, cv::Mat& gray) const;
  /// @brief Make image pyramid
  void MakePyramid(ImagePyramid& grays) const;

  /// @brief Track current frame
  bool TrackFrame();

  /// @brief Reinitialize if tracking failed
  void Reinitialize();

  /// @brief Whether we should add a keyframe
  bool ShouldAddKeyframe() const;
  /// @brief Add a new keyframe to window
  void AddKeyframe(const cv::Mat& depth);
  /// @brief Remove a keyframe from window
  void RemoveKeyframe();
  std::pair<int, double> FindKeyframeToRemove() const;
  /// @brief Optimize window
  void BundleAdjust();

  void Summarize(bool new_kf) const;
  void DrawFrame(const cv::Mat& depth) const;
  void DrawKeyframe() const;
};

}  // namespace sv::dsol

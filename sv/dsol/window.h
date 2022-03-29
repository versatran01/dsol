#pragma once

#include <absl/container/inlined_vector.h>

#include "sv/dsol/frame.h"

namespace sv::dsol {

/// @brief A sliding window of keyframes
/// @details Imagine we have 4 kfs, the window will allocate storage for 5,
/// where the last one will stores the previously removed kf.
/// Assuming we currently have 3 kfs in the window with max kf 4, and p points
/// to one past the last kf
///               p
/// w = [0, 1, 2, x, x]
/// After adding a new kf, the window is full, p points to one past the last
/// frame which is also the remove slot
///                  p
/// w = [0, 1, 2, 3, x]
/// We then remove kf 1, so it will be put into the remove slot
///               p
/// w = [0, 2, 3, x, 1]
/// And we add another new kf, the window is full again
///                  p
/// w = [0, 2, 3, 4, 1]
class KeyframeWindow {
 public:
  KeyframeWindow() = default;
  explicit KeyframeWindow(int num_kfs) { Resize(num_kfs); }
  KeyframeWindow(int num_kfs, int num_levels, const cv::Size& grid_size) {
    Allocate(num_kfs, num_levels, grid_size);
  }

  // Disable copy/move
  KeyframeWindow(const KeyframeWindow&) = delete;
  KeyframeWindow& operator=(const KeyframeWindow&) = delete;
  KeyframeWindow(KeyframeWindow&&) noexcept = delete;
  KeyframeWindow& operator=(KeyframeWindow&&) = delete;

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const KeyframeWindow& rhs) {
    return os << rhs.Repr();
  }

  /// @brief Clear window
  void Reset() noexcept { p_ = 0; }
  /// @brief Resize window, does not allocate
  void Resize(int num_kfs) noexcept;
  /// @brief Resize window and allocate keyframes
  size_t Allocate(int num_kfs,
                  int num_levels,
                  const cv::Size& grid_size) noexcept;

  /// @brief Info
  int size() const noexcept { return p_; }
  bool empty() const noexcept { return p_ == 0; }
  bool full() const noexcept { return p_ + 1 >= static_cast<int>(kfs_.size()); }
  int max_kfs() const noexcept { return static_cast<int>(kfs_.size()) - 1; }

  /// @brief Getters
  Keyframe& KfAt(int i);
  Keyframe& NextKf() { return KfAt(p_); }
  Keyframe& CurrKf() { return KfAt(p_ - 1); }

  const Keyframe& KfAt(int i) const;
  const Keyframe& NextKf() const { return KfAt(p_); }
  const Keyframe& CurrKf() const { return KfAt(p_ - 1); }
  const Keyframe& MargKf() const { return *ptrs_.back(); }

  KeyframePtrConstSpan keyframes() const noexcept {
    return absl::MakeConstSpan(ptrs_.data(), p_);
  }
  KeyframePtrSpan keyframes() noexcept {
    return absl::MakeSpan(ptrs_.data(), p_);
  }

  /// @brief Add keyframe from frame
  Keyframe& AddKeyframe(const Frame& frame);
  /// @brief Remove keyframe by index
  /// @return status of the removed kf
  KeyframeStatus RemoveKeyframeAt(int i);

  /// @brief Helper function to get all keyframe translations
  Eigen::Matrix3Xd GetAllTrans() const;
  Eigen::Matrix4Xd GetAllAffine() const;
  std::vector<Sophus::SE3d> GetAllPoses() const;

 private:
  int p_{};  // points to one past last frame
  //  std::vector<Keyframe*> ptrs_;
  absl::InlinedVector<Keyframe*, 8> ptrs_;
  std::vector<Keyframe> kfs_;
};

}  // namespace sv::dsol

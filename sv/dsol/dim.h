#pragma once

namespace sv::dsol {

/// @brief A struct that stores dimension info
struct Dim {
  static constexpr int kPoint = 1;   // inverse depth
  static constexpr int kPatch = 5;   // we use patch size of 5
  static constexpr int kAffine = 2;  // affine a and b
  static constexpr int kPose = 6;    // rot and trans
  static constexpr int kMono = kPose + kAffine;
  static constexpr int kStereo = kMono + kAffine;
  static constexpr int kFrame = kStereo;
};

}  // namespace sv::dsol

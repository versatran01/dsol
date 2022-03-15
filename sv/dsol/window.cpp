#include "sv/dsol/window.h"

#include "sv/util/logging.h"

namespace sv::dsol {

std::string KeyframeWindow::Repr() const {
  return fmt::format("KeyframeWindow(size={}/{})", size(), max_kfs());
}

void KeyframeWindow::Resize(int num_kfs) noexcept {
  // actually allocate num_kfs + 1 kf to store the marginalized kf
  ptrs_.resize(num_kfs + 1);
  kfs_.resize(num_kfs + 1);
  for (size_t i = 0; i < kfs_.size(); ++i) {
    ptrs_[i] = kfs_.data() + i;
  }
}

size_t KeyframeWindow::Allocate(int num_kfs,
                                int num_levels,
                                const cv::Size& grid_size) noexcept {
  Resize(num_kfs);
  size_t bytes = 0;
  for (auto& kf : kfs_) {
    bytes += kf.Allocate(num_levels, grid_size);
  }
  return bytes;
}

Keyframe& KeyframeWindow::KfAt(int i) {
  CHECK_LE(0, i);
  CHECK_LT(i, max_kfs());
  return *ptrs_.at(i);
}

const Keyframe& KeyframeWindow::KfAt(int i) const {
  CHECK_LE(0, i);
  CHECK_LT(i, max_kfs());
  return *ptrs_.at(i);
}

Keyframe& KeyframeWindow::AddKeyframe(const Frame& frame) {
  // Make sure window is not full and pointer is valid
  CHECK(!full());
  CHECK_LE(0, p_);

  auto& kf = NextKf();
  kf.SetFrame(frame);
  ++p_;
  return kf;
}

KeyframeStatus KeyframeWindow::RemoveKeyframeAt(int i) {
  CHECK_LE(0, i);
  CHECK_LT(i, size());

  // Copy status
  KeyframeStatus status = KfAt(i).status();

  // rotate left by 1 from this kf to last
  // Imagine we have 3 valid kfs [0, 1, 2] and we remove 1 (p points at 3)
  //     x     p
  // [0, 1, 2, 3, 4]
  // after rotate we have [0, 2] and 1 should be a t the end
  //        p
  // [0, 2, 3, 4, 1]
  // std::rotate(first, new_first, last)
  std::rotate(ptrs_.begin() + i, ptrs_.begin() + i + 1, ptrs_.end());
  --p_;
  return status;
}

Eigen::Matrix3Xd KeyframeWindow::GetAllTrans() const {
  Eigen::Matrix3Xd trans(3, size());
  for (int i = 0; i < size(); ++i) {
    trans.col(i) = KfAt(i).Twc().translation();
  }
  return trans;
}

Eigen::Matrix4Xd KeyframeWindow::GetAllAffine() const {
  Eigen::Matrix4Xd affine(4, size());
  for (int i = 0; i < size(); ++i) {
    const auto& state = KfAt(i).state();
    affine.col(i).head<2>() = state.affine_l.ab;
    affine.col(i).tail<2>() = state.affine_r.ab;
  }
  return affine;
}

std::vector<Sophus::SE3d> KeyframeWindow::GetAllPoses() const {
  std::vector<Sophus::SE3d> poses(size());
  for (int i = 0; i < size(); ++i) {
    poses.at(i) = KfAt(i).state().T_w_cl;
  }
  return poses;
}

}  // namespace sv::dsol

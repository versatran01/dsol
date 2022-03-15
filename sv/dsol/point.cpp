#include "sv/dsol/point.h"

#include "sv/util/logging.h"
#include "sv/dsol/pixel.h"

namespace sv::dsol {

void DepthPoint::SetIdepthInfo(double idepth, double info) noexcept {
  CHECK(PixelOk());
  CHECK_GE(idepth, 0);
  CHECK_LE(info, kMaxInfo);
  idepth_ = idepth;
  info_ = info;
}

std::string DepthPoint::Repr() const {
  return fmt::format("DepthPoint(uv=({},{}), idepth={:0.4f}, info={})",
                     px_.x,
                     px_.y,
                     idepth_,
                     info_);
}

// void FramePoint::UpdateJac() {
//  // n = Proj(p) = [x/z, y/z]^T
//  // p = [x, y, 1]^T
//  // dn_dp = [1/z,  0, -x/z^2] = [1, 0, -x] at [x, y, 1]
//  //         [0,  1/z, -y/z^2]   [0, 1, -y]
//  Matrix23d dn_dp;
//  dn_dp << 1, 0, -nc.x(), 0, 1, -nc.y();

//  // dn_dx = dn_dp * dp_dx
//  // dp_dx = [dp_dr, dp_dt]
//  // p  = dR * n0 / q0 + dt
//  // p' = dR * n       + dt * q0
//  // dp_dr = -[n]x
//  // dp_dt = q0 * I
//  dn_dx.leftCols<3>().noalias() = -dn_dp * Hat3d(nc.x(), nc.y(), 1);
//  dn_dx.rightCols<3>().noalias() = dn_dp * idepth_;
//}

/// ============================================================================
void Patch::ExtractAround(const cv::Mat& image,
                          const cv::Point2d& px) noexcept {
  for (int k = 0; k < kSize; ++k) {
    const auto px_k = px + kOffsetPx[k];
    vals[k] = ValAtD<uchar>(image, px_k);
    grads[k] = GradAtD<uchar>(image, px_k);
  }
}

void Patch::Extract(const cv::Mat& image, const Point2dArray& pxs) noexcept {
  for (int k = 0; k < Patch::kSize; ++k) {
    const auto& px = pxs[k];
    vals[k] = ValAtD<uchar>(image, px);
    grads[k] = GradAtD<uchar>(image, px);
  }
}

void Patch::ExtractFast(const cv::Mat& image,
                        const Point2dArray& pxs) noexcept {
  for (int k = 0; k < Patch::kSize; ++k) {
    // Use a specialied function to retieve grad/val with fewer accesses
    const auto xyv = GradValAtD<uchar>(image, pxs[k]);
    grads[k].x = xyv.x;
    grads[k].y = xyv.y;
    vals[k] = xyv.z;
  }
}

void Patch::ExtractIntensity(const cv::Mat& image,
                             const Point2dArray& pxs) noexcept {
  for (int k = 0; k < Patch::kSize; ++k) {
    vals[k] = ValAtD<uchar>(image, pxs[k]);
  }
}

bool Patch::IsAnyOut(const cv::Mat& mat, const Point2dArray& pxs) noexcept {
  return std::any_of(std::cbegin(pxs), std::cend(pxs), [&](const auto& px) {
    return IsPixOut(mat, px, kBorder);
  });
}

}  // namespace sv::dsol

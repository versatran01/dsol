#include "sv/dsol/camera.h"

#include <benchmark/benchmark.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <opencv2/highgui.hpp>

namespace sv::dsol {
namespace {

namespace bm = benchmark;

TEST(CameraTest, TestScaled) {
  const Camera cam({100, 100}, {10, 20, 30, 40}, 0);
  EXPECT_EQ(cam.Scaled(1).fxycxy().matrix(), cam.fxycxy().matrix());
  EXPECT_EQ(cam.Scaled(0.5).fxycxy().matrix(),
            Eigen::Array4d(5, 10, 14.75, 19.75).matrix());
  EXPECT_EQ(cam.Scaled(0.5).scale_, 0.5);
  EXPECT_EQ(cam.Scaled(0.5).Scaled(0.5).scale_, 0.25);
  EXPECT_EQ(cam.Scaled(0.5).Scaled(0.5).fxycxy().matrix(),
            cam.Scaled(0.25).fxycxy().matrix());
}

TEST(CameraTest, TestProject) {
  Eigen::Vector3d pt{8, 4, 2};
  Eigen::Vector2d nc = Project(pt);
  EXPECT_EQ(nc, Eigen::Vector2d(4, 2));

  Eigen::Matrix3d pts;
  pts << 8, 6, 4, 8, 6, 4, 4, 3, 2;
  const auto ncs = Project(pts);
  EXPECT_EQ(ncs.col(0), Eigen::Vector2d(2, 2));
  EXPECT_EQ(ncs.col(1), Eigen::Vector2d(2, 2));
  EXPECT_EQ(ncs.col(2), Eigen::Vector2d(2, 2));
}

TEST(CameraTest, TestPnormFromPixel) {
  const Eigen::Vector2d uv{8, 8};
  const Eigen::Array4d fc{2, 2, 4, 4};

  const auto nc = PnormFromPixel(uv, fc);
  EXPECT_EQ(nc, Eigen::Vector2d(2, 2));
  EXPECT_EQ(PixelFromPnorm(nc, fc), uv);

  Eigen::Matrix2d uvs;
  uvs << 4, 8, 4, 8;

  const auto ncs = PnormFromPixel(uvs, fc);
  EXPECT_EQ(ncs.col(0), Eigen::Vector2d(0, 0));
  EXPECT_EQ(ncs.col(1), Eigen::Vector2d(2, 2));
  EXPECT_EQ(PixelFromPnorm(ncs, fc), uvs);
}

TEST(CameraTest, TestScaledProjectBackProject) {
  const Eigen::Vector3d p(8, 4, 2);
  const Eigen::Array4d fc(4, 4, 2, 2);
  const double scale = 0.5;

  const auto nc = Project(p);

  // project then scale
  const auto uv = ScaleUv(PixelFromPnorm(nc, fc), scale);
  // project with scaled intrin
  const auto uv2 = PixelFromPnorm(nc, ScaleFxycxy(fc, scale));
  EXPECT_EQ(uv, uv2);
}

TEST(CameraTest, TestForward) {
  const Eigen::Vector3d pt(8, 4, 2);
  const Eigen::Array4d fc(4, 4, 2, 2);

  // project then apply intrinsics
  const auto uv = PixelFromPnorm(Project(pt), fc);

  const Camera camera{{}, fc, 0};
  const auto uv2 = camera.Forward(pt);
  EXPECT_EQ(uv, uv2);
}

TEST(CameraTest, TestBackward) {
  const Eigen::Vector2d uv(8, 8);
  const Eigen::Array4d fc(4, 4, 2, 2);

  // project then apply intrinsics
  const auto nch = Homogenize(PnormFromPixel(uv, fc));

  const Camera camera{{}, fc, 0};
  const auto nch2 = camera.Backward(uv);
  EXPECT_EQ(nch, nch2);
}

TEST(CameraTest, TestProjectJacobian) {
  const Eigen::Vector3d pt{4, 2, 2};
  MatrixMNd<2, 3> J;
  J << 0.5, 0, -1, 0, 0.5, -0.5;
  EXPECT_EQ(DprojDpoint(pt), J);
}

constexpr int N = 5;
// static const Eigen::Isometry3d kTf = Eigen::Isometry3d::Identity();
static const Eigen::Array4d kFc = Eigen::Array4d::Ones();

// TEST(CameraTest, TestWarp) {
//  const Eigen::Vector2d uv{4, 8};
//  const Eigen::Array4d fc{2, 2, 4, 4};
//  const double idepth = 1;

//  const auto uv1 = Warp(uv, fc, idepth, kTf, fc);
//  EXPECT_EQ(uv1, uv);

//  Eigen::Matrix<double, 2, 4> uvs;
//  uvs.setOnes();
//  const auto uvs1 = Warp(uvs, fc, idepth, kTf, fc);
//  EXPECT_EQ(uvs1, uvs);
//}

TEST(CameraTest, TestVignetteNoop) {
  const VignetteModel vm;
  EXPECT_TRUE(vm.Noop());
}

TEST(CameraTest, TestVignetteMap) {
  const VignetteModel vm(
      {1280, 800}, {640, 400}, {-0.345577, 0.486665, -0.375499});

  double min{};
  double max{};
  cv::minMaxIdx(vm.map(), &min, &max);

  LOG(INFO) << "min: " << min << ", max: " << max << "\n";
  EXPECT_EQ(max, 1.0);
  EXPECT_LT(min, max);

  //  cv::Mat x = vm.map().mul(vm.map());
  //  cv::imshow("vignette", x);
  //  cv::waitKey(-1);
}

/// ============================================================================
void BM_CameraForward(bm::State& state) {
  MatrixMNd<3, N> pts;
  pts.setOnes();
  const Camera camera{{}, kFc, 0};

  for (auto _ : state) {
    bm::DoNotOptimize(camera.Forward(pts));
  }
}
BENCHMARK(BM_CameraForward);

void BM_ProjectToPixel(bm::State& state) {
  MatrixMNd<3, N> pts;
  pts.setOnes();

  for (auto _ : state) {
    bm::DoNotOptimize(PixelFromPnorm(Project(pts), kFc));
  }
}
BENCHMARK(BM_ProjectToPixel);

void BM_CameraBackward(bm::State& state) {
  MatrixMNd<2, N> uvs;
  uvs.setOnes();
  const Camera camera{{}, kFc, 0};

  for (auto _ : state) {
    bm::DoNotOptimize(camera.Backward(uvs));
  }
}
BENCHMARK(BM_CameraBackward);

void BM_PnormHomogenize(bm::State& state) {
  MatrixMNd<2, N> uvs;
  uvs.setOnes();

  for (auto _ : state) {
    bm::DoNotOptimize(Homogenize(PnormFromPixel(uvs, kFc)));
  }
}
BENCHMARK(BM_PnormHomogenize);

void BM_VignetteUpdate(bm::State& state) {
  VignetteModel vm({640, 640}, {320, 320}, {1, 1, 1});

  const int gsize = static_cast<int>(state.range(0));
  for (auto _ : state) {
    vm.UpdateMap(gsize);
    bm::DoNotOptimize(vm.map_);
  }
}
BENCHMARK(BM_VignetteUpdate)->Arg(0)->Arg(1);

}  // namespace
}  // namespace sv::dsol

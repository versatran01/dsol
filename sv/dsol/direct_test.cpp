#include "sv/dsol/direct.h"

#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

namespace sv::dsol {
namespace {

namespace bm = benchmark;

TEST(DirectTest, TestTransformScaled) {
  Eigen::Vector3d p0{1, 1, 1};
  Eigen::Isometry3d T0 = Eigen::Isometry3d::Identity();

  const auto p1 = TransformScaled(T0, p0, 1);
  EXPECT_EQ(p1, p0);

  const auto p2 = TransformScaled(T0, p0, 2);
  EXPECT_EQ(p2, p0);

  Eigen::Isometry3d T1 = Eigen::Isometry3d::Identity();
  T1.translation() = Eigen::Vector3d::Ones();

  const auto p3 = TransformScaled(T1, p0, 1);
  EXPECT_EQ(p3, Eigen::Vector3d(2, 2, 2));

  const auto p4 = TransformScaled(T1, p0, 2);
  EXPECT_EQ(p4, Eigen::Vector3d(3, 3, 3));
}

TEST(DirectSolveCfgTest, TestGetNumLevels) {
  DirectSolveCfg cfg{};
  cfg.max_levels = 0;
  EXPECT_EQ(cfg.GetNumLevels(4), 4);

  cfg.max_levels = 2;
  EXPECT_EQ(cfg.GetNumLevels(4), 2);

  cfg.max_levels = 5;
  EXPECT_EQ(cfg.GetNumLevels(4), 4);
}

TEST(DirectCostCfgTest, TestGetFrameDim) {
  DirectCostCfg cfg{};
  cfg.affine = false;
  cfg.stereo = false;
  EXPECT_EQ(cfg.GetFrameDim(), Dim::kPose);

  cfg.affine = false;
  cfg.stereo = true;
  EXPECT_EQ(cfg.GetFrameDim(), Dim::kPose);

  cfg.affine = true;
  cfg.stereo = false;
  EXPECT_EQ(cfg.GetFrameDim(), Dim::kMono);

  cfg.affine = true;
  cfg.stereo = true;
  EXPECT_EQ(cfg.GetFrameDim(), Dim::kStereo);
}

/// ============================================================================
const Eigen::Isometry3d kTf = Eigen::Isometry3d::Identity();
const Eigen::Array4d kFc = Eigen::Array4d::Ones();
constexpr int N = 5;

void BM_WarpSingle(bm::State& state) {
  const Eigen::Vector2d uv{1, 1};

  for (auto _ : state) {
    const auto uv1 = Warp(uv, kFc, 1, kTf, kFc);
    bm::DoNotOptimize(uv1);
  }
}
BENCHMARK(BM_WarpSingle);

void BM_WarpSequential(bm::State& state) {
  MatrixMNd<2, N> uv0s;
  uv0s.setOnes();
  MatrixMNd<2, N> uv1s;

  for (auto _ : state) {
    for (int i = 0; i < N; ++i) {
      uv1s.col(i) = Warp(uv0s.col(i).eval(), kFc, 1, kTf, kFc);
    }
    bm::DoNotOptimize(uv1s);
  }
}
BENCHMARK(BM_WarpSequential);

void BM_WarpSep(bm::State& state) {
  MatrixMNd<2, N> uv0s;
  uv0s.setOnes();
  MatrixMNd<2, N> uv1s;

  for (auto _ : state) {
    uv1s.col(0) = Warp(uv0s.col(0).eval(), kFc, 1.0, kTf, kFc);
    uv1s.rightCols<N - 1>() =
        Warp(uv0s.rightCols<N - 1>().eval(), kFc, 1.0, kTf, kFc);
    bm::DoNotOptimize(uv1s);
  }
}
BENCHMARK(BM_WarpSep);

void BM_WarpBatch(bm::State& state) {
  MatrixMNd<2, N> uv0s;
  uv0s.setOnes();
  MatrixMNd<2, N> uv1s;

  for (auto _ : state) {
    uv1s = Warp(uv0s, kFc, 1.0, kTf, kFc);
    bm::DoNotOptimize(uv1s);
  }
}
BENCHMARK(BM_WarpBatch);

void BM_WarpBatchCamera(bm::State& state) {
  MatrixMNd<2, N> uv0s;
  uv0s.setOnes();
  const Camera camera{{}, kFc, 0};
  MatrixMNd<2, N> uv1s;

  for (auto _ : state) {
    const auto nh0s = camera.Backward(uv0s);
    const auto pt1s = TransformScaled(kTf, nh0s, 1);
    uv1s = camera.Forward(pt1s);
    bm::DoNotOptimize(uv1s);
  }
}
BENCHMARK(BM_WarpBatchCamera);

}  // namespace
}  // namespace sv::dsol

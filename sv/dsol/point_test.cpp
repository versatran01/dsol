#include "sv/dsol/point.h"

#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

namespace sv::dsol {
namespace {

namespace bm = benchmark;

TEST(PointTest, TestUpdate) {
  DepthPoint p;
  p.SetPix({1, 1});

  p.SetIdepthInfo(1, 1);
  p.UpdateIdepth(0.5);
  EXPECT_EQ(p.idepth(), 1.5);
  EXPECT_TRUE(p.DepthOk());

  p.UpdateIdepth(-2000);
  EXPECT_EQ(p.idepth(), 0.0);
  EXPECT_TRUE(p.DepthOk());

  p.UpdateInfo(0.5);
  EXPECT_EQ(p.info(), 1.5);
  EXPECT_TRUE(!p.InfoOk());
  p.UpdateInfo(1000);
  EXPECT_EQ(p.info(), DepthPoint::kMaxInfo);
  EXPECT_TRUE(p.InfoOk());
  EXPECT_TRUE(p.InfoMax());
}

void BM_JacobianMul(bm::State& state) {
  Eigen::Matrix<double, 2, 3> A;
  A.setRandom();
  Eigen::Matrix<double, 3, 6> B;
  B.setRandom();

  for (auto _ : state) {
    Eigen::Matrix<double, 2, 6> C = A * B;
    bm::DoNotOptimize(C);
  }
}
BENCHMARK(BM_JacobianMul);

void BM_JacobianMul2(bm::State& state) {
  Eigen::Matrix<double, 2, 4> A;
  A.setRandom();
  Eigen::Matrix<double, 4, 6> B;
  B.setRandom();

  for (auto _ : state) {
    Eigen::Matrix<double, 2, 6> C = A * B;
    bm::DoNotOptimize(C);
  }
}
BENCHMARK(BM_JacobianMul2);

}  // namespace
}  // namespace sv::dsol

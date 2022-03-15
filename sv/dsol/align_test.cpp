#include "sv/dsol/align.h"

#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

namespace sv::dsol {
namespace {

TEST(AlignCostTest, TestBuildHessian) {
  FrameHessian1 hess;
  AlignCost cost;
  Patch patch;
  patch.vals.setZero();
  FramePoint point;
  Eigen::Vector2d nc{1, 1};

  EXPECT_TRUE(cost.UpdateHess(patch, point, patch, hess));
  EXPECT_TRUE(cost.UpdateHess(patch, point, patch, hess));
}

namespace bm = benchmark;

void BM_AlignCostUpdateHess(bm::State& state) {
  FrameHessian1 hess;
  AlignCost cost;
  Patch patch;
  patch.vals.setZero();
  FramePoint point;

  for (auto _ : state) {
    const auto ok = cost.UpdateHess(patch, point, patch, hess);
    bm::DoNotOptimize(ok);
  }
}
BENCHMARK(BM_AlignCostUpdateHess);

void BM_AlignCostUpdateHessNoAffine(bm::State& state) {
  FrameHessian1 hess;
  AlignCost cost;
  cost.cfg.affine = false;  // no affine
  Patch patch;
  patch.vals.setZero();
  FramePoint point;
  Eigen::Vector2d nc{1, 1};

  for (auto _ : state) {
    const auto ok = cost.UpdateHess(patch, point, patch, hess);
    bm::DoNotOptimize(ok);
  }
}
BENCHMARK(BM_AlignCostUpdateHessNoAffine);

}  // namespace
}  // namespace sv::dsol

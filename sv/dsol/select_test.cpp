#include "sv/dsol/select.h"

#include <benchmark/benchmark.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>

namespace sv::dsol {
namespace {

namespace bm = benchmark;

constexpr int kNumLevels = 2;
constexpr int kImageSize = 640;
constexpr int kCellSize = 16;
const cv::Size kGridSize = {kImageSize / kCellSize, kImageSize / kCellSize};

TEST(SelectTest, TestFindMaxGrad) {
  // clang-format off
  // (2,2) is the max grad point
  const cv::Mat image =
      (cv::Mat_<uchar>(4, 4) << 1, 1, 1, 1,
                                1, 2, 9, 1,
                                0, 5, 7, 0,
                                0, 0, 0, 0);
  // clang-format on
  const auto pxg = FindMaxGrad(image, {1, 1, 2, 2});
  EXPECT_EQ(pxg.px.x, 2);
  EXPECT_EQ(pxg.px.y, 2);
}

TEST(SelectTest, TestAllocate) {
  PixelSelector selector;
  const cv::Size top_size{160, 320};  // 10 x 20
  const cv::Size sel_size{80, 160};
  EXPECT_EQ(selector.Allocate(top_size, sel_size), 19200);
}

/// ============================================================================
void BM_SelectLevel0(bm::State& state) {
  ImagePyramid images;
  MakeImagePyramid(MakeRandMat8U(kImageSize), kNumLevels, images);

  SelectCfg cfg;
  cfg.sel_level = 0;
  cfg.max_grad = 256;
  PixelSelector det{cfg};

  const auto gsize = static_cast<int>(state.range(0));
  for (auto _ : state) {
    const auto n = det.Select(images, gsize);
    bm::DoNotOptimize(n);
  }
}
BENCHMARK(BM_SelectLevel0)->Arg(0)->Arg(1);

void BM_SelectLevel1(bm::State& state) {
  ImagePyramid images;
  MakeImagePyramid(MakeRandMat8U(kImageSize), kNumLevels, images);

  SelectCfg cfg;
  cfg.max_grad = 256;
  PixelSelector det{cfg};

  const auto gsize = static_cast<int>(state.range(0));
  for (auto _ : state) {
    const auto n = det.Select(images, gsize);
    bm::DoNotOptimize(n);
  }
}
BENCHMARK(BM_SelectLevel1)->Arg(0)->Arg(1);

}  // namespace
}  // namespace sv::dsol

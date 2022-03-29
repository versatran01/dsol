#include "sv/dsol/pixel.h"

#include <benchmark/benchmark.h>
#include <gtest/gtest.h>

namespace sv::dsol {
namespace {

namespace bm = benchmark;

// This is actually slower by 1ns
TEST(PixelTest, TestValAtD) {
  const cv::Mat image = (cv::Mat_<uchar>(2, 2) << 1, 3, 7, 13);
  // [1] -  2 -  [3]
  //  |     |     |
  //  4 -   6  -  8
  //  |     |     |
  // [7] - 10 -  [13]

  EXPECT_EQ(ValAtD<uchar>(image, {0, 0}), 1);
  EXPECT_EQ(ValAtD<uchar>(image, {1, 0}), 3);
  EXPECT_EQ(ValAtD<uchar>(image, {0, 1}), 7);
  EXPECT_EQ(ValAtD<uchar>(image, {1, 1}), 13);

  EXPECT_EQ(ValAtD<uchar>(image, {0.25, 0}), 1.5);
  EXPECT_EQ(ValAtD<uchar>(image, {0.5, 0}), 2);
  EXPECT_EQ(ValAtD<uchar>(image, {0.75, 0}), 2.5);
  EXPECT_EQ(ValAtD<uchar>(image, {0.5, 1}), 10);
  EXPECT_EQ(ValAtD<uchar>(image, {0, 0.5}), 4);
  EXPECT_EQ(ValAtD<uchar>(image, {1, 0.5}), 8);

  EXPECT_EQ(ValAtD<uchar>(image, {0.5, 0.5}), 6);
}

TEST(PixelTest, TestValAtD2) {
  const cv::Mat image =
      (cv::Mat_<uchar>(4, 4) << 0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0);

  EXPECT_EQ(ValAtD<uchar>(image, {1, 1}), 1);
  EXPECT_EQ(ValAtD<uchar>(image, {2, 1}), 2);
  EXPECT_EQ(ValAtD<uchar>(image, {1, 2}), 3);
  EXPECT_EQ(ValAtD<uchar>(image, {2, 2}), 4);

  EXPECT_EQ(ValAtD<uchar>(image, {1.5, 1}), 1.5);
  EXPECT_EQ(ValAtD<uchar>(image, {1.5, 2}), 3.5);
  EXPECT_EQ(ValAtD<uchar>(image, {1, 1.5}), 2);
  EXPECT_EQ(ValAtD<uchar>(image, {2, 1.5}), 3);

  EXPECT_EQ(ValAtD<uchar>(image, {1.5, 1.5}), 2.5);
}

TEST(PixelTest, TestValAtE) {
  const cv::Mat image = (cv::Mat_<uchar>(2, 2) << 1, 3, 7, 13);
  // [1] -  2 -  [3]
  //  |     |     |
  //  4 -   6  -  8
  //  |     |     |
  // [7] - 10 -  [13]

  EXPECT_EQ(ValAtE<uchar>(image, {0, 0}), 1);
  EXPECT_EQ(ValAtE<uchar>(image, {1, 0}), 3);
  EXPECT_EQ(ValAtE<uchar>(image, {0, 1}), 7);
  EXPECT_EQ(ValAtE<uchar>(image, {1, 1}), 13);

  EXPECT_EQ(ValAtE<uchar>(image, {0.25, 0}), 1.5);
  EXPECT_EQ(ValAtE<uchar>(image, {0.5, 0}), 2);
  EXPECT_EQ(ValAtE<uchar>(image, {0.75, 0}), 2.5);
  EXPECT_EQ(ValAtE<uchar>(image, {0.5, 1}), 10);
  EXPECT_EQ(ValAtE<uchar>(image, {0, 0.5}), 4);
  EXPECT_EQ(ValAtE<uchar>(image, {1, 0.5}), 8);

  EXPECT_EQ(ValAtE<uchar>(image, {0.5, 0.5}), 6);
}

TEST(PixelTest, TestGradAt) {
  const cv::Mat image = (cv::Mat_<uchar>(3, 3) << 1, 2, 4, 3, 6, 9, 4, 8, 12);
  // 1 2 4
  // 3 6 9
  // 4 8 12
  EXPECT_EQ(GradXAtI<uchar>(image, {1, 0}), 1.5);
  EXPECT_EQ(GradXAtI<uchar>(image, {1, 1}), 3);
  EXPECT_EQ(GradXAtI<uchar>(image, {1, 2}), 4);

  EXPECT_EQ(GradYAtI<uchar>(image, {0, 1}), 1.5);
  EXPECT_EQ(GradYAtI<uchar>(image, {1, 1}), 3);
  EXPECT_EQ(GradYAtI<uchar>(image, {2, 1}), 4);

  EXPECT_EQ(GradXAtD<uchar>(image, {1, 1}), 3);
  EXPECT_EQ(GradYAtD<uchar>(image, {1, 1}), 3);
}

TEST(PixelTest, TestGradValAtD) {
  const cv::Mat image = (cv::Mat_<uchar>(2, 2) << 1, 3, 7, 13);
  // [1] -  2 -  [3]
  //  |     |     |
  //  4 -   6  -  8
  //  |     |     |
  // [7] - 10 -  [13]

  EXPECT_EQ(GradValAtD<uchar>(image, {0, 0}), cv::Point3d(2, 6, 1));
  EXPECT_EQ(GradValAtD<uchar>(image, {0.25, 0}), cv::Point3d(2, 7, 1.5));
  EXPECT_EQ(GradValAtD<uchar>(image, {0.5, 0}), cv::Point3d(2, 8, 2));
  EXPECT_EQ(GradValAtD<uchar>(image, {0.5, 0.5}), cv::Point3d(4, 8, 6));
}

TEST(PixelTest, TestIsPixOut) {
  EXPECT_TRUE(IsPixOut({10, 10}, {-1, -1}));
  EXPECT_TRUE(IsPixOut({10, 10}, {10, 10}));
  EXPECT_TRUE(!IsPixOut({10, 10}, {0, 0}));
  EXPECT_TRUE(!IsPixOut({10, 10}, {9, 9}));

  EXPECT_TRUE(IsPixOut({10, 10}, {0, 0}, 1));
  EXPECT_TRUE(IsPixOut({10, 10}, {9, 9}, 1));
  EXPECT_TRUE(!IsPixOut({10, 10}, {1, 1}, 1));
  EXPECT_TRUE(!IsPixOut({10, 10}, {8, 8}, 1));
}

TEST(PixelTest, TestScalePix) {
  EXPECT_EQ(ScalePix({10, 10}, 1), cv::Point2d(10, 10));
  EXPECT_EQ(ScalePix({10, 10}, 0.5), cv::Point2d(4.75, 4.75));
}

/// ============================================================================
constexpr int kSize = 320;
constexpr int kHalfSize = kSize / 2;
const Eigen::Vector2d kUv = {kHalfSize, kHalfSize};
const cv::Point2d kPx = {kHalfSize, kHalfSize};
const cv::Mat kImage = cv::Mat::ones(kSize, kSize, CV_8UC1);

void BM_ValAtD(bm::State& state) {
  for (auto _ : state) {
    bm::DoNotOptimize(ValAtD<uchar>(kImage, kPx));
  }
}
BENCHMARK(BM_ValAtD);

void BM_ValAtE(bm::State& state) {
  for (auto _ : state) {
    bm::DoNotOptimize(ValAtE<uchar>(kImage, kUv));
  }
}
BENCHMARK(BM_ValAtE);

void BM_GradAtD(bm::State& state) {
  for (auto _ : state) {
    bm::DoNotOptimize(GradAtD<uchar>(kImage, kPx));
  }
}
BENCHMARK(BM_GradAtD);

void BM_GradAtE(bm::State& state) {
  for (auto _ : state) {
    bm::DoNotOptimize(GradAtE<uchar>(kImage, kUv));
  }
}
BENCHMARK(BM_GradAtE);

void BM_ValGradAtD(bm::State& state) {
  for (auto _ : state) {
    bm::DoNotOptimize(GradValAtD<uchar>(kImage, kPx));
  }
}
BENCHMARK(BM_ValGradAtD);

void BM_ValGradAtE(bm::State& state) {
  for (auto _ : state) {
    bm::DoNotOptimize(GradValAtE<uchar>(kImage, kUv));
  }
}
BENCHMARK(BM_ValGradAtE);

void BM_ValGradAtDSep(bm::State& state) {
  for (auto _ : state) {
    bm::DoNotOptimize(ValAtD<uchar>(kImage, kPx));
    bm::DoNotOptimize(GradAtD<uchar>(kImage, kPx));
  }
}
BENCHMARK(BM_ValGradAtDSep);

void BM_SobelAtI(bm::State& state) {
  for (auto _ : state) {
    bm::DoNotOptimize(SobelAtI<uchar>(kImage, kPx));
  }
}
BENCHMARK(BM_SobelAtI);

}  // namespace
}  // namespace sv::dsol

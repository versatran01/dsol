#include "sv/dsol/image.h"

#include <benchmark/benchmark.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>

namespace sv::dsol {
namespace {

namespace bm = benchmark;

TEST(ImageTest, TestCropImageFactor) {
  cv::Mat x(123, 321, CV_8UC1);
  const auto c = CropImageFactor(x, 32);
  EXPECT_EQ(c.rows, 96);
  EXPECT_EQ(c.cols, 320);
}

TEST(ImageTest, TestCvSaturation) {
  cv::Mat x = cv::Mat::zeros(1, 1, CV_8UC1);
  x += 256;
  EXPECT_EQ(x.at<uchar>(0), 255);
  x -= 256;
  EXPECT_EQ(x.at<uchar>(0), 0);
}

TEST(ImageTest, TestMakeImagePyramid) {
  cv::Mat image(123, 321, CV_8UC1);
  ImagePyramid images;
  MakeImagePyramid(image, 4, images);
  EXPECT_TRUE(IsImagePyramid(images));

  images[0] = cv::Mat::zeros(111, 222, CV_8UC1);
  EXPECT_TRUE(!IsImagePyramid(images));
}

/// ============================================================================
void BM_MakeImagePyramid(bm::State& state) {
  const auto image = MakeRandMat8U(state.range(0));
  ImagePyramid images;
  for (auto _ : state) {
    MakeImagePyramid(image, 4, images);
    bm::DoNotOptimize(images);
  }
}
BENCHMARK(BM_MakeImagePyramid)->Arg(256)->Arg(512)->Arg(1024);

}  // namespace
}  // namespace sv::dsol

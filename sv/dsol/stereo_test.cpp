#include "sv/dsol/stereo.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

namespace sv::dsol {

TEST(StereoTest, TestZeroMeanNormalize) {
  Eigen::Array3d v{1, 2, 3};
  ZeroMeanNormalize(v);
  EXPECT_DOUBLE_EQ(v.mean(), 0);
  EXPECT_DOUBLE_EQ(v.matrix().norm(), 1);
}

TEST(StereoTest, TestExtractRoiArrayXf) {
  const cv::Mat mat = (cv::Mat_<uchar>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);

  Eigen::Array3f arr;
  ExtractRoiArrayXf(mat, {0, 1, 3, 1}, arr);
  EXPECT_EQ(arr.matrix(), Eigen::Vector3f(4, 5, 6));

  ExtractRoiArrayXf(mat, {1, 0, 1, 3}, arr);
  EXPECT_EQ(arr.matrix(), Eigen::Vector3f(2, 5, 8));
}

TEST(StereoTest, TestAllocate) {
  StereoMatcher matcher;
  EXPECT_EQ(matcher.Allocate({10, 20}), 10 * 20 * 2);
}

}  // namespace sv::dsol

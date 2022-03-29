#include "sv/util/ocv.h"

#include <gtest/gtest.h>

namespace sv {
namespace {

TEST(OcvTest, TestCvTypeStr) {
  EXPECT_EQ(CvTypeStr(CV_8U), "8UC1");
  EXPECT_EQ(CvTypeStr(CV_8S), "8SC1");
  EXPECT_EQ(CvTypeStr(CV_16U), "16UC1");
  EXPECT_EQ(CvTypeStr(CV_16S), "16SC1");
  EXPECT_EQ(CvTypeStr(CV_32S), "32SC1");
  EXPECT_EQ(CvTypeStr(CV_32F), "32FC1");
  EXPECT_EQ(CvTypeStr(CV_64F), "64FC1");

  EXPECT_EQ(CvTypeStr(CV_8UC1), "8UC1");
  EXPECT_EQ(CvTypeStr(CV_8UC2), "8UC2");
  EXPECT_EQ(CvTypeStr(CV_8UC3), "8UC3");
  EXPECT_EQ(CvTypeStr(CV_8UC4), "8UC4");
  EXPECT_EQ(CvTypeStr(CV_8SC1), "8SC1");
  EXPECT_EQ(CvTypeStr(CV_8SC2), "8SC2");
  EXPECT_EQ(CvTypeStr(CV_8SC3), "8SC3");
  EXPECT_EQ(CvTypeStr(CV_8SC4), "8SC4");
  EXPECT_EQ(CvTypeStr(CV_32FC1), "32FC1");
  EXPECT_EQ(CvTypeStr(CV_32FC2), "32FC2");
  EXPECT_EQ(CvTypeStr(CV_32FC3), "32FC3");
  EXPECT_EQ(CvTypeStr(CV_32FC4), "32FC4");
}

TEST(OcvTest, TestRange) {
  cv::Range r{1, 2};
  auto s = r * 2;
  EXPECT_EQ(s.start, 2);
  EXPECT_EQ(s.end, 4);

  auto d = s / 2;
  EXPECT_EQ(d.start, 1);
  EXPECT_EQ(d.end, 2);
}

}  // namespace
}  // namespace sv

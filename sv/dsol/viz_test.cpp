#include "sv/dsol/viz.h"

#include <gtest/gtest.h>

namespace sv::dsol {
namespace {

TEST(DisplayTest, TestCtor) {
  const auto image = MakeRandMat8U(640);
  ImagePyramid images;
  MakeImagePyramid(image, 4, images);

  PyramidDisplay disp{images};

  EXPECT_EQ(disp.levels(), images.size());
  for (int i = 0; i < disp.levels(); ++i) {
    const auto& im0 = images.at(i);
    const auto& im1 = disp.LevelAt(i);
    EXPECT_EQ(im0.rows, im1.rows);
    EXPECT_EQ(im0.cols, im1.cols);
  }
}

}  // namespace
}  // namespace sv::dsol

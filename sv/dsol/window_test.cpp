#include "sv/dsol/window.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace sv::dsol {
namespace {

class KeyframeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ImagePyramid images;
    MakeImagePyramid(MakeRandMat8U(640), 4, images);
    frame = Frame(images, {});
  }

  Frame frame;
};

TEST_F(KeyframeTest, TestDefaultCtor) {
  KeyframeWindow win;

  EXPECT_TRUE(win.empty());
  EXPECT_TRUE(win.full());
  EXPECT_EQ(win.size(), 0);
  EXPECT_EQ(win.max_kfs(), -1);
}

TEST_F(KeyframeTest, TestSizeCtor) {
  KeyframeWindow win(4);

  EXPECT_TRUE(win.empty());
  EXPECT_FALSE(win.full());
  EXPECT_EQ(win.size(), 0);
  EXPECT_EQ(win.max_kfs(), 4);
}

TEST_F(KeyframeTest, TestAddKeyframe) {
  KeyframeWindow win(2);
  win.AddKeyframe(frame);

  EXPECT_EQ(win.empty(), false);
  EXPECT_EQ(win.full(), false);
  EXPECT_EQ(win.size(), 1);

  win.AddKeyframe(frame);
  EXPECT_EQ(win.empty(), false);
  EXPECT_EQ(win.full(), true);
  EXPECT_EQ(win.size(), 2);

  // This should fail
  //  win.AddKeyframe(frame);
}

TEST_F(KeyframeTest, TestRemoveKeyframe) {
  KeyframeWindow win(4);
  win.AddKeyframe(frame);
  win.AddKeyframe(frame);
  win.AddKeyframe(frame);

  EXPECT_EQ(win.size(), 3);
  EXPECT_EQ(win.full(), false);

  win.RemoveKeyframeAt(1);
  EXPECT_EQ(win.size(), 2);
}

TEST(RotateTest, TestStableRotate) {
  {
    // Starting with 3 elements [0, 1, 2], p is at 3
    //     x     p
    // [0, 1, 2, 3, 4]
    std::vector<int> x = {0, 1, 2, 3, 4};

    // Want to remove 1 such that we have [0, 2] left
    int i = 1;
    std::rotate(x.begin() + i, x.begin() + i + 1, x.end());

    //        p
    // [0, 2, 3, 4, 1]
    ASSERT_THAT(x, ::testing::ElementsAre(0, 2, 3, 4, 1));
  }

  {
    //        x  p
    // [0, 1, 2, 3, 4]
    std::vector<int> x = {0, 1, 2, 3, 4};
    int i = 2;
    std::rotate(x.begin() + i, x.begin() + i + 1, x.end());

    //        p
    // [0, 1, 3, 4, 2]
    ASSERT_THAT(x, ::testing::ElementsAre(0, 1, 3, 4, 2));
  }

  {
    //  x        p
    // [0, 1, 2, 3, 4]
    std::vector<int> x = {0, 1, 2, 3, 4};
    int i = 0;
    std::rotate(x.begin() + i, x.begin() + i + 1, x.end());

    //        p
    // [1, 2, 3, 4, 0]
    ASSERT_THAT(x, ::testing::ElementsAre(1, 2, 3, 4, 0));
  }
}

}  // namespace
}  // namespace sv::dsol

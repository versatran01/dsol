#include "sv/util/grid2d.h"

#include <gtest/gtest.h>

namespace sv {
namespace {

TEST(Grid2dTest, TestCtor) {
  Grid2d<int> g;
  EXPECT_EQ(g.size(), 0);
  EXPECT_EQ(g.area(), 0);
  EXPECT_EQ(g.empty(), true);
}

TEST(Grid2dTest, TestCtorSize) {
  Grid2d<int> g(2, 3);
  EXPECT_EQ(g.size(), 6);
  EXPECT_EQ(g.area(), 6);
  EXPECT_EQ(g.empty(), false);
}

}  // namespace
}  // namespace sv

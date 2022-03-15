#include "sv/util/tbb.h"

#include <gtest/gtest.h>

namespace sv {
namespace {

TEST(TbbTest, TestBlockedRange) {
  const BlockedRange range{0, 10, -1};
  EXPECT_EQ(range.begin_, 0);
  EXPECT_EQ(range.end_, 10);
  EXPECT_EQ(range.gsize_, 10);

  const BlockedRange range2{0, 10, 0};
  EXPECT_EQ(range2.begin_, 0);
  EXPECT_EQ(range2.end_, 10);
  EXPECT_EQ(range2.gsize_, 10);

  const BlockedRange range3{0, 10, 1};
  EXPECT_EQ(range3.begin_, 0);
  EXPECT_EQ(range3.end_, 10);
  EXPECT_EQ(range3.gsize_, 1);
}

}  // namespace
}  // namespace sv

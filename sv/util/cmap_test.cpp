#include "sv/util/cmap.h"

#include <gtest/gtest.h>

namespace sv {

TEST(CmapTest, TestJet) {
  const auto jet = MakeCmapJet();

  EXPECT_EQ(jet.name(), "jet");
  EXPECT_EQ(jet.GetRgb(-1), ColorMap::Rgb(0.0, 0.0, 0.5));
  EXPECT_EQ(jet.GetRgb(0), ColorMap::Rgb(0.0, 0.0, 0.5));
  EXPECT_EQ(jet.GetRgb(1), ColorMap::Rgb(0.5, 0.0, 0.0));
  EXPECT_EQ(jet.GetRgb(2), ColorMap::Rgb(0.5, 0.0, 0.0));
}

}  // namespace sv

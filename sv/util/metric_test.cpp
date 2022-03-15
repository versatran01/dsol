#include "sv/util/metric.h"

#include <fmt/ranges.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

namespace sv {

TEST(MetircTest, TestDefault) {
  DepthMetrics dm;
  EXPECT_TRUE(dm.empty());
  const auto metrics = dm.Comptue();
  EXPECT_TRUE(dm.empty());
}

TEST(MetircTest, TestUpdate) {
  DepthMetrics dm;

  dm.Update(2, 5);
  EXPECT_EQ(dm.num(), 1);
  auto ms = dm.Comptue();

  EXPECT_EQ(ms.at("mae"), 3);
  EXPECT_EQ(ms.at("rmse"), 3);
  EXPECT_EQ(ms.at("absrel"), 1.5);
  EXPECT_EQ(ms.at("sqrel"), 4.5);

  dm.Update(2, 6);
  EXPECT_EQ(dm.num(), 2);
  ms = dm.Comptue();
  EXPECT_EQ(ms.at("mae"), 3.5);
  EXPECT_DOUBLE_EQ(ms.at("rmse"), std::sqrt(12.5));
  EXPECT_EQ(ms.at("absrel"), 1.75);
  EXPECT_EQ(ms.at("sqrel"), 6.25);

  dm.Reset();
  EXPECT_EQ(dm.num(), 0);

  LOG(INFO) << fmt::format("{}", ms);
}

}  // namespace sv

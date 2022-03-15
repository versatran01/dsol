#pragma once

#include <absl/container/btree_map.h>

namespace sv {

using MetricDict = absl::btree_map<std::string, double>;

struct DepthMetrics {
  int n_{0};
  MetricDict data_;

  DepthMetrics() = default;

  int num() const noexcept { return n_; }
  bool empty() const noexcept { return n_ == 0; }

  void Update(double gt, double pred);
  void Reset();
  MetricDict Comptue();
};

}  // namespace sv

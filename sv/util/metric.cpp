#include "sv/util/metric.h"

#include "sv/util/logging.h"

namespace sv {

void DepthMetrics::Update(double gt, double pred) {
  CHECK_GE(gt, 0);
  CHECK_GE(pred, 0);

  const auto err = gt - pred;
  const auto err_abs = std::abs(err);
  const auto err_sq = err * err;

  data_["mae"] += err_abs;
  data_["rmse"] += err_sq;
  data_["absrel"] += err_abs / gt;
  data_["sqrel"] += err_sq / gt;

  ++n_;
}

void DepthMetrics::Reset() {
  n_ = 0;
  data_.clear();
}

MetricDict DepthMetrics::Comptue() {
  MetricDict metrics;
  if (n_ == 0) return metrics;

  for (const auto& kv : data_) {
    metrics[kv.first] = kv.second / n_;
  }

  metrics["rmse"] = std::sqrt(metrics.at("rmse"));
  return metrics;
}

}  // namespace sv

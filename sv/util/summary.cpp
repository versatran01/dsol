#include "sv/util/summary.h"

#include <fmt/ostream.h>

#include "sv/util/logging.h"

namespace sv {

std::string StatsSummary::ReportStats(const std::string& name,
                                        const StatsT& stats) const {
  std::string str = fmt::format(fmt::fg(fmt::color::cyan), "[{:<16}]", name);

  str += fmt::format(
      " n: {:<8} | mean: {:<14.4e} | last: {:<14.4f} | min: {:<14.4f} | "
      "max: {:<14.4f} | sum: {:<14.4f} |",
      stats.count(),
      stats.mean(),
      stats.last(),
      stats.min(),
      stats.max(),
      stats.sum());
  return str;
}

TimerSummary::ManualTimer::ManualTimer(std::string name,
                                         TimerSummary* manager,
                                         bool start)
    : name_{std::move(name)}, manager_{CHECK_NOTNULL(manager)} {
  if (start) {
    timer_.Start();
  } else {
    timer_.Reset();
  }
}

void TimerSummary::ManualTimer::Stop(bool record) {
  timer_.Stop();
  if (record) {
    stats_.Add(absl::Nanoseconds(timer_.Elapsed()));
  }
}

void TimerSummary::ManualTimer::Commit() {
  Stop(true);
  if (!stats_.ok()) return;  // Noop if there's no stats to commit

  // Already checked in ctor
  // CHECK_NOTNULL(manager_);
  manager_->Update(name_, stats_);
  stats_ = StatsT{};  // reset stats
}

std::string TimerSummary::ReportStats(const std::string& name,
                                        const StatsT& stats) const {
  std::string str =
      fmt::format(fmt::fg(fmt::color::light_sky_blue), "[{:<16}]", name);
  str += fmt::format(
      " n: {:<8} | last: {:<14} | mean: {:<14} | min: {:<14} | max: {:<14} | "
      "sum: {:<14} |",
      stats.count(),
      stats.last(),
      stats.mean(),
      stats.min(),
      stats.max(),
      stats.sum());
  return str;
}

}  // namespace sv

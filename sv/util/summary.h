#pragma once

#include <absl/container/flat_hash_map.h>
#include <absl/time/time.h>

#include <mutex>
#include <shared_mutex>
#include <string_view>

#include "sv/util/stats.h"
#include "sv/util/timer.h"

/// Specialize numeric_limits for absl::Duration (follows integral type)
/// https://codeyarns.com/tech/2015-07-02-max-min-and-lowest-in-c.html
/// https://quuxplusone.github.io/blog/2021/10/27/dont-reopen-namespace-std/
template <>
struct std::numeric_limits<absl::Duration> {
  static constexpr bool is_specialized = true;
  static constexpr absl::Duration min() { return absl::ZeroDuration(); }
  static constexpr absl::Duration lowest() { return absl::ZeroDuration(); }
  static constexpr absl::Duration max() { return absl::InfiniteDuration(); }
};

namespace sv {

/// This is similar to Ceres Solver's ExecutionSummary class, where we record
/// execution statistics (mainly time). Instead of simply record the time, we
/// store a bunch of other useful statistics, like min, max, mean, etc.
template <typename T>
class SummaryBase {
 public:
  using StatsT = Stats<T>;

  explicit SummaryBase(std::string name = "stats") : name_{std::move(name)} {}
  virtual ~SummaryBase() noexcept = default;

  bool empty() const noexcept { return stats_dict_.empty(); }
  auto size() const noexcept { return stats_dict_.size(); }
  const auto& name() const noexcept { return name_; }
  const auto& dict() const { return stats_dict_; }

  /// @brief Merge stats given name, create a new one if name is new.
  /// @details Thread-safe
  /// @note Use string_view because absl::*_map supports heterogenous lookup
  void Merge(std::string_view name, const StatsT& stats) {
    if (!stats.ok()) return;
    std::unique_lock lock{mutex_};
    stats_dict_[name] += stats;
  }

  /// @brief Add new data to stats given name, create a new one if name is new.
  /// @details Thread-safe
  void Add(std::string_view name, const T& val) {
    std::unique_lock lock{mutex_};
    stats_dict_[name].Add(val);
  }

  /// @brief Get stats under name
  /// @details Thread-safe
  /// @return Copy of stats if found, otherwise returns empty stats
  StatsT GetStats(std::string_view name) const {
    std::shared_lock lock{mutex_};
    const auto it = stats_dict_.find(name);
    if (it != stats_dict_.end()) return it->second;
    lock.unlock();  // Not found, unlock
    return {};
  }

  /// @brief Return a string of all stats
  std::string ReportAll(bool sort = false) const {
    std::string str = "Manager: " + name_;

    std::shared_lock lock{mutex_};
    if (sort) {
      std::vector<std::string> keys;
      keys.reserve(stats_dict_.size());
      for (const auto& kv : stats_dict_) keys.push_back(kv.first);
      std::sort(keys.begin(), keys.end());
      for (const auto& key : keys) {
        str += "\n" + ReportStats(key, stats_dict_.at(key));
      }
    } else {
      for (const auto& kv : stats_dict_) {
        str += "\n" + ReportStats(kv.first, kv.second);
      }
    }
    return str;
  }

  /// @brief Return a string of a stats by name
  std::string Report(const std::string& name) const {
    return ReportStats(name_ + "/" + std::string(name), GetStats(name));
  }

  virtual std::string ReportStats(const std::string& name,
                                  const StatsT& stats) const = 0;

 protected:
  using StatsDict = absl::flat_hash_map<std::string, StatsT>;

  std::string name_;                 // name of the manager
  StatsDict stats_dict_;             // use a dict to store stats
  mutable std::shared_mutex mutex_;  // reader-writer mutex
};

/// A simple stats manager of type double
class StatsSummary final : public SummaryBase<double> {
 public:
  using SummaryBase::SummaryBase;

  /// @brief Get stats by name
  StatsT& GetRef(std::string_view name) { return stats_dict_[name]; }

  std::string ReportStats(const std::string& name,
                          const StatsT& stats) const override;
};

/// This is similar to Ceres Solver's ExecutionSummary class, where we record
/// execution statistics (mainly time). Instead of simply record the time, we
/// store a bunch of other useful statistics, like min, max, mean, etc.
class TimerSummary final : public SummaryBase<absl::Duration> {
 public:
  /// A manual timer where user needs to call stop and commit explicitly
  /// Create multiple timers if you want to log time in multiple-threads
  class ManualTimer {
   public:
    /// If start is true then start on construction, otherise stopped
    ManualTimer(std::string name, TimerSummary* manager, bool start = true);
    virtual ~ManualTimer() noexcept = default;

    /// Disable copy, allow move
    ManualTimer(const ManualTimer&) = delete;
    ManualTimer& operator=(const ManualTimer&) = delete;
    ManualTimer(ManualTimer&&) noexcept = default;
    ManualTimer& operator=(ManualTimer&&) = default;

    /// Start the timer
    void Start() { timer_.Start(); }

    /// Stop and record the elapsed time after Start()
    void Stop(bool record = true);

    /// Resume stopped timer
    void Resume() { timer_.Resume(); }

    /// Commit changes to manager, potentially expensive since it needs to
    /// acquire a lock
    void Commit();

   private:
    Timer timer_;            // actual timer
    StatsT stats_;           // local stats
    std::string name_;       // name of timer
    TimerSummary* manager_;  // ref to manager
  };

  /// A scoped timer that will call stop on destruction
  class ScopedTimer : public ManualTimer {
   public:
    using ManualTimer::ManualTimer;

    /// Disable copy and move
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;
    ScopedTimer(ScopedTimer&&) noexcept = default;
    ScopedTimer& operator=(ScopedTimer&&) = default;
    ~ScopedTimer() override { Commit(); }
  };

  explicit TimerSummary(const std::string& name = "timers")
      : SummaryBase{name} {}

  /// Start a ManualTimer by name, need to manually stop the returned timer.
  /// Elapsed time will automatically added to the stats when stopped.
  /// After stop one can just call timer.Start() to restart.
  /// Need to call Commit() to aggregate stats
  ManualTimer Manual(std::string name, bool start = true) {
    return {std::move(name), this, start};
  }

  /// Returns a ScopedTimer (already started) and will stop when out of scope
  ScopedTimer Scoped(std::string name) { return {std::move(name), this}; }

  /// Return a string of timer statistics
  std::string ReportStats(const std::string& name,
                          const StatsT& stats) const override;
};

}  // namespace sv

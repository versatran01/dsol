#pragma once

#include <absl/time/clock.h>

#include <chrono>
#include <cstdint>

namespace sv {

/// @brief A simple Timer class, not thread-safe
class Timer {
 public:
  /// Timer starts right away
  Timer() { Start(); }

  bool IsRunning() const noexcept { return running_; }
  bool IsStopped() const noexcept { return !running_; }
  //  static int64_t NowNs() { return absl::GetCurrentTimeNanos(); }
  static int64_t NowNs() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(
               high_resolution_clock::now().time_since_epoch())
        .count();
  }

  /// Start timer, repeated calls to Start() will update the start time
  void Start() {
    running_ = true;
    time_ns_ = NowNs();  // time_ns_ is when the timer started
  }

  /// Stop timer, repeated calls to Stop() have no effect after the first one
  void Stop() {
    if (!running_) return;
    running_ = false;
    time_ns_ = NowNs() - time_ns_;  // time_ns_ is elapsed time since start
  }

  /// Resume timer, keep counting from the last Stop()
  /// Noop if the timer is already running
  /// repeated calls to Resume() have no effect
  void Resume() {
    if (running_) return;
    const auto prev_elapsed = time_ns_;
    Start();                   // time_ns is now the new start time
    time_ns_ -= prev_elapsed;  // subtract prev_elapsed from new start time
  }

  /// Return elapsed time in nanoseconds, will not stop timer
  int64_t Elapsed() const {
    if (!running_) return time_ns_;  // time_ns_ is already elapsed
    return NowNs() - time_ns_;       // time_ns_ is when the timer started
  }

  /// Reset timer
  void Reset() noexcept {
    time_ns_ = 0;
    running_ = false;
  }

 private:
  int64_t time_ns_{0};  // either start (running) or elapsed (stopped)
  bool running_{false};
};

}  // namespace sv

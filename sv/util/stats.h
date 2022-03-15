#pragma once

#include <algorithm>  // min/max
#include <limits>     // numeric_limits

namespace sv {

template <typename T>
class Stats {
 public:
  using value_type = T;

  void Add(const T& value) noexcept {
    ++count_;
    sum_ += value;
    min_ = std::min(min_, value);
    max_ = std::max(max_, value);
    last_ = value;
  }

  bool ok() const noexcept { return count_ > 0; }
  int count() const noexcept { return count_; }
  T sum() const noexcept { return sum_; }
  T min() const noexcept { return min_; }
  T max() const noexcept { return max_; }
  T last() const noexcept { return last_; }
  T mean() const noexcept { return count_ > 0 ? sum_ / count_ : T{}; }

  Stats<T>& operator+=(const Stats<T>& rhs) noexcept {
    if (rhs.count() > 0) {
      count_ += rhs.count();
      sum_ += rhs.sum();
      min_ = std::min(min_, rhs.min());
      max_ = std::max(max_, rhs.max());
      last_ = rhs.last();
    }
    return *this;
  }

  friend Stats<T> operator+(Stats<T> lhs, const Stats<T>& rhs) noexcept {
    return lhs += rhs;
  }

 private:
  int count_{0};
  T sum_{};
  T min_{std::numeric_limits<T>::max()};
  T max_{std::numeric_limits<T>::lowest()};
  T last_{};
};

using StatsF = Stats<float>;
using StatsD = Stats<double>;

}  // namespace sv

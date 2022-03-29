#pragma once

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

namespace sv {

/// @brief Simplified BlockedRange, gsize <= 0 will only use single thread
struct BlockedRange {
  BlockedRange() = default;
  BlockedRange(int begin, int end, int gsize)
      : begin_{begin}, end_{end}, gsize_{gsize <= 0 ? end - begin : gsize} {}
  int begin_{};
  int end_{};
  int gsize_{};

  auto ToTbb() const noexcept {
    return tbb::blocked_range<int>(begin_, end_, gsize_);
  }
};

/// @brief Wrapper for tbb::parallel_reduce
template <typename T, typename F, typename R>
T ParallelReduce(const BlockedRange& range,
                 const T& identity,
                 const F& function,
                 const R& reduction) {
  return tbb::parallel_reduce(
      range.ToTbb(),
      identity,
      [&](const auto& block, T local) {
        for (int i = block.begin(); i < block.end(); ++i) {
          function(i, local);
        }
        return local;
      },
      reduction);
}

/// @brief Wrapper for tbb::parallel_for
template <typename F>
void ParallelFor(const BlockedRange& range, const F& function) {
  tbb::parallel_for(range.ToTbb(), [&](const auto& block) {
    for (int i = block.begin(); i < block.end(); ++i) {
      function(i);
    }
  });
}

}  // namespace sv

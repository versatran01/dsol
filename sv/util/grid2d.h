#pragma once

#include <glog/logging.h>

#include <opencv2/core/types.hpp>

namespace sv {

template <typename T>
class Grid2d {
 public:
  // export vector types
  using container = std::vector<T>;
  using value_type = typename container::value_type;
  using pointer = typename container::pointer;
  using const_pointer = typename container::const_pointer;
  using referece = typename container::reference;
  using const_referece = typename container::const_reference;
  using iterator = typename container::iterator;
  using const_iterator = typename container::const_iterator;
  using const_reverse_iterator = typename container::const_reverse_iterator;
  using reverse_iterator = typename container::reverse_iterator;
  using size_type = typename container::size_type;
  using difference_type = typename container::difference_type;
  using allocator_type = typename container::allocator_type;

  Grid2d() = default;
  Grid2d(int rows, int cols, const T& val = {})
      : grid_size_{cols, rows}, data_(grid_size_.area(), val) {}
  explicit Grid2d(const cv::Size& cvsize, const T& val = {})
      : Grid2d{cvsize.height, cvsize.width, val} {}

  void reset(const T& val = {}) { data_.assign(size(), val); }

  void resize(const cv::Size& cvsize, const T& val = {}) {
    grid_size_ = cvsize;
    data_.resize(grid_size_.area(), val);
  }

  T& at(int r, int c) { return data_.at(rc2ind(r, c)); }
  const T& at(int r, int c) const { return data_.at(rc2ind(r, c)); }

  T& at(cv::Point2i pt) { return at(pt.y, pt.x); }
  const T& at(cv::Point2i pt) const { return at(pt.y, pt.x); }

  T& at(size_t i) { return data_.at(i); }
  const T& at(size_t i) const { return data_.at(i); }

  cv::Size cvsize() const noexcept { return grid_size_; }
  int area() const noexcept { return grid_size_.area(); }
  bool empty() const noexcept { return data_.empty(); }
  size_t size() const noexcept { return data_.size(); }

  int cols() const noexcept { return grid_size_.width; }
  int rows() const noexcept { return grid_size_.height; }
  int width() const noexcept { return grid_size_.width; }
  int height() const noexcept { return grid_size_.height; }

  int rc2ind(int r, int c) const noexcept { return r * cols() + c; }

  iterator begin() noexcept { return data_.begin(); }
  iterator end() noexcept { return data_.end(); }

  const_iterator begin() const noexcept { return data_.begin(); }
  const_iterator end() const noexcept { return data_.end(); }

  const_iterator cbegin() const { return data_.cbegin(); }
  const_iterator cend() const noexcept { return data_.cend(); }

 private:
  cv::Size grid_size_{};  // actual grid size
  std::vector<T> data_{};
};

}  // namespace sv

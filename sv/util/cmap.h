#pragma once

#include <Eigen/Core>
#include <string>

namespace sv {

/// @brief A simple color map implementation inspired by
/// github.com/yuki-koyama/tinycolormap/blob/master/include/tinycolormap.hpp
class ColorMap {
 public:
  using Rgb = Eigen::Vector3d;

  ColorMap() = default;
  ColorMap(std::string name, const std::vector<Eigen::Vector3d>& colors);

  std::string Repr() const;
  friend std::ostream& operator<<(std::ostream& os, const ColorMap& rhs) {
    return os << rhs.Repr();
  }

  /// @brief Map input x to color rgb/bgr, assumes x is in [0, 1]
  Eigen::Vector3d GetRgb(double x) const noexcept;
  Eigen::Vector3d GetBgr(double x) const noexcept {
    return GetRgb(x).reverse();
  }

  bool Ok() const noexcept { return !data_.empty(); }
  const std::string& name() const noexcept { return name_; }
  int size() const noexcept { return static_cast<int>(data_.size()); }

 private:
  std::string name_;
  double step_{};
  std::vector<Eigen::Vector3d> data_;
};

/// @brief Factory
ColorMap MakeCmapJet();
ColorMap MakeCmapHeat();
ColorMap MakeCmapTurbo();
ColorMap MakeCmapPlasma();
ColorMap GetColorMap(std::string_view name);

}  // namespace sv

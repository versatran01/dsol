#pragma once

#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <glog/logging.h>

namespace sv {

struct LogColor {
  using term_color = fmt::terminal_color;
  static constexpr auto kRed = fmt::fg(term_color::red);
  static constexpr auto kGreen = fmt::fg(term_color::green);
  static constexpr auto kYellow = fmt::fg(term_color::yellow);
  static constexpr auto kBlue = fmt::fg(term_color::blue);
  static constexpr auto kMagenta = fmt::fg(term_color::magenta);
  static constexpr auto kCyan = fmt::fg(term_color::cyan);

  static constexpr auto kBrightRed = fmt::fg(term_color::bright_red);
  static constexpr auto kBrightGreen = fmt::fg(term_color::bright_green);
  static constexpr auto kBrightYellow = fmt::fg(term_color::bright_yellow);
  static constexpr auto kBrightBlue = fmt::fg(term_color::bright_blue);
  static constexpr auto kBrightMagenta = fmt::fg(term_color::bright_magenta);
  static constexpr auto kBrightCyan = fmt::fg(term_color::bright_cyan);
};

}  // namespace sv

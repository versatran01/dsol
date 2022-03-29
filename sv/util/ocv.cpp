#include "sv/util/ocv.h"

#include "sv/util/logging.h"

namespace sv {

std::string CvTypeStr(int type) noexcept {
  cv::Mat a;
  std::string r;

  const uchar depth = type & CV_MAT_DEPTH_MASK;
  const auto chans = static_cast<uchar>(1 + (type >> CV_CN_SHIFT));

  switch (depth) {
    case CV_8U:
      r = "8U";
      break;
    case CV_8S:
      r = "8S";
      break;
    case CV_16U:
      r = "16U";
      break;
    case CV_16S:
      r = "16S";
      break;
    case CV_32S:
      r = "32S";
      break;
    case CV_32F:
      r = "32F";
      break;
    case CV_64F:
      r = "64F";
      break;
    default:
      r = "User";
      break;
  }

  return fmt::format("{}C{}", r, chans);
}

std::string Repr(const cv::Mat& mat) {
  return fmt::format("(hwc=({},{},{}), depth={})",
                     mat.rows,
                     mat.cols,
                     mat.channels(),
                     mat.depth());
}

std::string Repr(const cv::Size& size) {
  return fmt::format("(h={}, w={})", size.height, size.width);
}

std::string Repr(const cv::Range& range) {
  return fmt::format("[{},{})", range.start, range.end);
}

void Imshow(const std::string& name,
            const cv::Mat& mat,
            int flag,
            cv::Point offset) {
  cv::namedWindow(name, flag);
  cv::moveWindow(name, offset.x, offset.y);
  cv::imshow(name, mat);
}

cv::Mat ApplyCmap(const cv::Mat& input,
                  double scale,
                  int cmap,
                  uint8_t bad_color) {
  CHECK_EQ(input.channels(), 1);

  cv::Mat disp;
  input.convertTo(disp, CV_8UC1, scale * 255.0);
  cv::applyColorMap(disp, disp, cmap);

  if (input.depth() >= CV_32F) {
    disp.setTo(bad_color, cv::Mat(~(input > 0)));
  }

  return disp;
}

/// ============================================================================
namespace {

constexpr char kKeyEsc = 27;
constexpr char kKeySpace = 32;
constexpr char kKeyP = 112;
constexpr char kKeyR = 114;
constexpr char kKeyS = 115;
constexpr bool kKill = false;
constexpr bool kAlive = true;

void WriteText(cv::Mat& image,
               const std::string& text,
               cv::HersheyFonts font = cv::FONT_HERSHEY_DUPLEX) {
  cv::putText(image,
              text,
              {0, 24},                    // org
              font,                       // font
              1.0,                        // scale
              cv::Scalar(255, 255, 255),  // color
              2,                          // thick
              cv::LINE_AA);
}

}  // namespace

KeyControl::KeyControl(int wait_ms, const cv::Size& size)
    : paused_{wait_ms > 0}, wait_ms_{wait_ms} {
  display_ = cv::Mat(size, CV_8UC3, paused_ ? color_pause_ : color_run_);
  if (paused_) {
    WriteText(display_, std::to_string(counter_));
  } else {
    WriteText(display_, "RUNNING");
  }

  if (wait_ms_ > 0) {
    LOG(INFO)
        << "Press 's' to step, 'r' to play, 'p' to pause, 'space' to toggle "
           "play/pause, 'esc' to quit";
    cv::namedWindow(name_);
    cv::imshow(name_, display_);
    cv::moveWindow(name_, 0, 0);
  }
}

bool KeyControl::Wait() {
  ++counter_;

  if (wait_ms_ <= 0) {
    return kAlive;
  }

  while (true) {
    if (paused_) {
      auto key = cv::waitKey(0);  // Pause
      switch (key) {
        case kKeySpace:
          // [space] while stop pause and return immediately
          display_ = color_run_;
          WriteText(display_, "RUNNING");
          cv::imshow(name_, display_);
          paused_ = false;
          return kAlive;
        case kKeyS:
          // [s] will return but keep pausing
          display_ = color_step_;
          WriteText(display_, std::to_string(counter_));
          cv::imshow(name_, display_);
          paused_ = true;
          return kAlive;
        case kKeyEsc:
          return kKill;
      }
    } else {
      // not pause, so it is running
      const auto key = cv::waitKey(wait_ms_);
      // Press space or P will pause
      if (key == kKeySpace || key == kKeyP) {
        display_ = color_pause_;
        WriteText(display_, std::to_string(counter_));
        cv::imshow(name_, display_);
        paused_ = true;
      } else {
        if (key == kKeyEsc) {
          return kAlive;
        }
        break;
      }
    }
  }

  return kAlive;
}

/// ============================================================================
WindowTiler::WindowTiler(const cv::Size& screen_size,
                         const cv::Point& offset,
                         const cv::Point& start)
    : screen_size_{screen_size}, offset_{offset}, start_{start}, curr_{start} {}

void WindowTiler::Tile(const std::string& name, const cv::Mat& mat, int flag) {
  cv::namedWindow(name, flag);
  cv::moveWindow(name, curr_.x, curr_.y);
  cv::imshow(name, mat);
  Next();
}

void WindowTiler::Next() noexcept {
  if (curr_.x + offset_.x > screen_size_.width) {
    curr_.x = 0;
    if (curr_.y + offset_.y > screen_size_.height) {
      curr_.y = 0;
    } else {
      curr_.y += offset_.y;
    }
  } else {
    curr_.x += offset_.x;
  }
}

}  // namespace sv

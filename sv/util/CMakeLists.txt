cc_library(
  NAME util_log
  HDRS "logging.h"
  DEPS sv_base fmt::fmt glog::glog
  DEFINES FMT_DEPRECATED_OSTREAM
  INTERFACE)

cc_library(
  NAME util_grid2d
  HDRS "grid2d.h"
  DEPS sv_base opencv_core
  INTERFACE)
cc_test(
  NAME util_grid2d_test
  SRCS "grid2d_test.cpp"
  DEPS sv_util_grid2d benchmark::benchmark)
cc_bench(
  NAME util_grid2d_bench
  SRCS "grid2d_test.cpp"
  DEPS sv_util_grid2d GTest::GTest)

cc_library(
  NAME util_eigen
  SRCS "eigen.cpp"
  DEPS sv_util_log Eigen3::Eigen)
cc_test(
  NAME util_eigen_test
  SRCS "eigen_test.cpp"
  DEPS sv_util_eigen benchmark::benchmark)
cc_bench(
  NAME util_eigen_bench
  SRCS "eigen_test.cpp"
  DEPS sv_util_eigen GTest::GTest)

cc_library(
  NAME util_cmap
  SRCS "cmap.cpp"
  DEPS sv_util_log Eigen3::Eigen)
cc_test(
  NAME util_cmap_test
  SRCS "cmap_test.cpp"
  DEPS sv_util_cmap)

cc_library(
  NAME util_tbb
  HDRS "tbb.h"
  DEPS sv_base TBB::tbb
  INTERFACE)
cc_test(
  NAME util_tbb_test
  SRCS "tbb_test.cpp"
  DEPS sv_util_tbb)

cc_library(
  NAME util_timer
  HDRS "timer.h"
  DEPS sv_base absl::time
  INTERFACE)

cc_library(
  NAME util_summary
  SRCS "summary.cpp"
  DEPS sv_util_log sv_util_timer absl::flat_hash_map)

cc_library(
  NAME util_math
  SRCS "math.cpp"
  DEPS sv_util_eigen)
cc_test(
  NAME util_math_test
  SRCS "math_test.cpp"
  DEPS sv_util_math benchmark::benchmark)
cc_bench(
  NAME util_math_bench
  SRCS "math_test.cpp"
  DEPS sv_util_math GTest::GTest)

cc_library(
  NAME util_ocv
  SRCS "ocv.cpp"
  DEPS sv_util_log opencv_core opencv_highgui opencv_imgproc)
cc_test(
  NAME util_ocv_test
  SRCS "ocv_test.cpp"
  DEPS sv_util_ocv)

cc_library(
  NAME util_dataset
  SRCS "dataset.cpp"
  DEPS sv_util_ocv Eigen3::Eigen Sophus::Sophus absl::flat_hash_map
  DEFINES XTENSOR_FOUND=${xtensor_FOUND})
if(xtensor_FOUND)
  target_link_libraries(sv_util_dataset PUBLIC xtensor)
endif()
cc_binary(
  NAME util_dataset_play
  SRCS "dataset_play.cpp"
  DEPS sv_util_dataset absl::flags_parse)

cc_library(
  NAME util_metric
  SRCS "metric.cpp"
  DEPS sv_util_log absl::btree)
cc_test(
  NAME util_metric_test
  SRCS "metric_test.cpp"
  DEPS sv_util_metric)

cc_library(
  NAME dsol_pixel
  HDRS "pixel.h"
  DEPS sv_base opencv_core Eigen3::Eigen
  INTERFACE)
cc_test(
  NAME dsol_pixel_test
  SRCS "pixel_test.cpp"
  DEPS sv_dsol_pixel benchmark::benchmark)
cc_bench(
  NAME dsol_pixel_bench
  SRCS "pixel_test.cpp"
  DEPS sv_dsol_pixel GTest::GTest)

cc_library(
  NAME dsol_image
  SRCS "image.cpp"
  DEPS sv_util_log opencv_core
  LINKOPTS opencv_imgproc)
cc_test(
  NAME dsol_image_test
  SRCS "image_test.cpp"
  DEPS sv_dsol_image benchmark::benchmark)
cc_bench(
  NAME dsol_image_bench
  SRCS "image_test.cpp"
  DEPS sv_dsol_image GTest::GTest)

cc_library(
  NAME dsol_camera
  SRCS "camera.cpp"
  DEPS sv_util_eigen opencv_core
  LINKOPTS sv_util_tbb)
cc_test(
  NAME dsol_camera_test
  SRCS "camera_test.cpp"
  DEPS sv_dsol_camera benchmark::benchmark opencv_highgui)
cc_bench(
  NAME dsol_camera_bench
  SRCS "camera_test.cpp"
  DEPS sv_dsol_camera GTest::GTest opencv_highgui)

cc_library(
  NAME dsol_point
  SRCS "point.cpp"
  DEPS sv_dsol_pixel sv_util_eigen sv_util_grid2d)
cc_test(
  NAME dsol_point_test
  SRCS "point_test.cpp"
  DEPS sv_dsol_point benchmark::benchmark)
cc_bench(
  NAME dsol_point_bench
  SRCS "point_test.cpp"
  DEPS sv_dsol_point GTest::GTest)

cc_library(
  NAME dsol_select
  SRCS "select.cpp"
  DEPS sv_dsol_image sv_dsol_point absl::span
  LINKOPTS sv_util_tbb)
cc_test(
  NAME dsol_select_test
  SRCS "select_test.cpp"
  DEPS sv_dsol_select benchmark::benchmark)
cc_bench(
  NAME dsol_select_bench
  SRCS "select_test.cpp"
  DEPS sv_dsol_select GTest::GTest)

cc_library(
  NAME dsol_frame
  SRCS "frame.cpp"
  DEPS sv_dsol_image sv_dsol_point sv_dsol_camera Sophus::Sophus absl::span
  LINKOPTS sv_util_tbb)
cc_test(
  NAME dsol_frame_test
  SRCS "frame_test.cpp"
  DEPS sv_dsol_frame benchmark::benchmark)
cc_bench(
  NAME dsol_frame_bench
  SRCS "frame_test.cpp"
  DEPS sv_dsol_frame GTest::GTest)

cc_library(
  NAME dsol_stereo
  SRCS "stereo.cpp"
  DEPS sv_dsol_frame
  LINKOPTS sv_util_tbb)
cc_test(
  NAME dsol_stereo_test
  SRCS "stereo_test.cpp"
  DEPS sv_dsol_stereo)

cc_library(
  NAME dsol_window
  SRCS "window.cpp"
  DEPS sv_dsol_frame)
cc_test(
  NAME dsol_window_test
  SRCS "window_test.cpp"
  DEPS sv_dsol_window)

cc_library(
  NAME dsol_hessian
  SRCS "hessian.cpp"
  DEPS sv_util_eigen opencv_core
  LINKOPTS sv_util_tbb)
cc_test(
  NAME dsol_hessian_test
  SRCS "hessian_test.cpp"
  DEPS sv_dsol_hessian benchmark::benchmark)
cc_bench(
  NAME dsol_hessian_bench
  SRCS "hessian_test.cpp"
  DEPS sv_dsol_hessian GTest::GTest)

cc_library(
  NAME dsol_solve
  SRCS "solve.cpp"
  DEPS sv_util_eigen)
cc_test(
  NAME dsol_solve_test
  SRCS "solve_test.cpp"
  DEPS sv_dsol_solve)

cc_library(
  NAME dsol_direct
  SRCS "direct.cpp"
  DEPS sv_dsol_camera sv_dsol_frame)
cc_test(
  NAME dsol_direct_test
  SRCS "direct_test.cpp"
  DEPS sv_dsol_direct benchmark::benchmark)
cc_bench(
  NAME dsol_direct_bench
  SRCS "direct_test.cpp"
  DEPS sv_dsol_direct GTest::GTest)

cc_library(
  NAME dsol_align
  SRCS "align.cpp"
  DEPS sv_dsol_hessian sv_dsol_direct sv_dsol_solve
  LINKOPTS sv_util_tbb)
cc_bench(
  NAME dsol_align_bench
  SRCS "align_test.cpp"
  DEPS sv_dsol_align GTest::GTest)

cc_library(
  NAME dsol_adjust
  SRCS "adjust.cpp"
  DEPS sv_dsol_hessian sv_dsol_direct
  LINKOPTS sv_util_tbb)

cc_library(
  NAME dsol_viz
  SRCS "viz.cpp"
  DEPS sv_dsol_image sv_dsol_point sv_util_cmap sv_util_math
  LINKOPTS sv_util_ocv)
cc_test(
  NAME dsol_viz_test
  SRCS "viz_test.cpp"
  DEPS sv_dsol_viz)

cc_library(
  NAME dsol_extra
  SRCS "extra.cpp"
  DEPS sv_dsol_frame sv_dsol_select sv_util_dataset sv_util_summary)

cc_library(
  NAME dsol_odom
  SRCS "odom.cpp"
  DEPS sv_dsol_adjust
       sv_dsol_align
       sv_dsol_select
       sv_dsol_stereo
       sv_dsol_window
       sv_dsol_viz
       sv_util_summary
       sv_util_ocv)
cc_binary(
  NAME dsol_odom_play
  SRCS "odom_play.cpp"
  DEPS sv_dsol_odom sv_dsol_extra sv_util_dataset absl::flags_parse)

if(BUILD_PLAY)
  cc_binary(
    NAME dsol_select_play
    SRCS "select_play.cpp"
    DEPS sv_dsol_select sv_dsol_viz sv_dsol_extra absl::flags_parse)

  cc_binary(
    NAME dsol_stereo_play
    SRCS "stereo_play.cpp"
    DEPS sv_dsol_stereo sv_dsol_select sv_dsol_extra sv_dsol_viz sv_util_metric
         absl::flags_parse)

  cc_binary(
    NAME dsol_align_play
    SRCS "align_play.cpp"
    DEPS sv_dsol_window sv_dsol_select sv_dsol_align sv_dsol_viz sv_dsol_extra
         absl::flags_parse)

  cc_binary(
    NAME dsol_adjust_play
    SRCS "adjust_play.cpp"
    DEPS sv_dsol_window sv_dsol_adjust sv_dsol_select sv_dsol_viz sv_dsol_extra
         absl::flags_parse)
endif()

if(BUILD_ROS)
  cc_library(
    NAME dsol_node_util
    SRCS "node_util.cpp"
    DEPS sv_ros1_msg_conv sv_dsol_direct sv_dsol_odom)

  cc_binary(
    NAME dsol_node_data
    SRCS "node_data.cpp"
    DEPS sv_dsol_extra sv_util_dataset sv_dsol_node_util)

  cc_binary(
    NAME dsol_node_odom
    SRCS "node_odom.cpp"
    DEPS sv_dsol_extra sv_util_dataset sv_dsol_node_util)
endif()

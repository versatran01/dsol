cc_library(
  NAME ros1
  DEPS ${catkin_LIBRARIES}
  INCS ${catkin_INCLUDE_DIRS}
  INTERFACE)

cc_library(
  NAME ros1_msg_conv
  SRCS "msg_conv.cpp"
  DEPS sv_base sv_ros1 Eigen3::Eigen Sophus::Sophus)

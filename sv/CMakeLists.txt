cc_library(
  NAME base
  DEPS sv_options
  INCS ${CMAKE_SOURCE_DIR}
  INTERFACE)

if(BUILD_VECTORIZE)
  target_compile_options(sv_base INTERFACE "-march=native")
endif()

add_subdirectory(util)
add_subdirectory(dsol)

if(BUILD_ROS)
  add_subdirectory(ros1)
endif()

# cmake-format: off
# cc_library()
#
# CMake function to imitate Bazel's cc_library rule.
#
# Parameters:
# NAME: name of target (see Note)
# HDRS: List of public header files for the library
# SRCS: List of source files for the library
# DEPS: List of other libraries to be linked in to the binary targets
# COPTS: List of private compile options
# DEFINES: List of public definitions
# LINKOPTS: List of link options
#
# Note:
# 
# By default, cc_library will always create a library named 
# ${CC_TARGET_PREFIX}_${NAME}, and alias target ${CC_TARGET_PREFIX}::${NAME}.  
# The ${CC_TARGET_PREFIX}:: form should always be used.
# This is to reduce namespace pollution.
#
# cc_library(
#   NAME
#     awesome
#   HDRS
#     "a.h"
#   SRCS
#     "a.cc"
# )
# cc_library(
#   NAME
#     fantastic_lib
#   SRCS
#     "b.cc"
#   DEPS
#     sv::awesome # not "awesome" !
# )
#
# cc_library(
#   NAME
#     main_lib
#   ...
#   DEPS
#     sv::fantastic_lib
# )
# cmake-format: on
function(cc_library)
  cmake_parse_arguments(CC_LIB "INTERFACE" "NAME"
                        "HDRS;SRCS;COPTS;DEFINES;LINKOPTS;DEPS;INCS" ${ARGN})

  if(CC_TARGET_PREFIX)
    set(_NAME "${CC_TARGET_PREFIX}_${CC_LIB_NAME}")
  else()
    set(_NAME ${CC_LIB_NAME})
  endif()

  # Check if this is a header-only library Note that as of February 2019, many
  # popular OS's (for example, Ubuntu 16.04 LTS) only come with cmake 3.5 by
  # default.  For this reason, we can't use list(FILTER...)
  set(SV_SRCS "${CC_LIB_SRCS}")
  foreach(src_file IN LISTS SV_SRCS)
    if(${src_file} MATCHES ".*\\.(h|hpp|inc)")
      list(REMOVE_ITEM SV_SRCS "${src_file}")
    endif()
  endforeach()

  if(CC_LIB_INTERFACE)
    set(CC_LIB_IS_INTERFACE 1)
  else()
    set(CC_LIB_IS_INTERFACE 0)
  endif()

  if(NOT CC_LIB_IS_INTERFACE)
    add_library(${_NAME} "")
    target_sources(${_NAME} PRIVATE ${CC_LIB_SRCS} ${CC_LIB_HDRS})
    target_link_libraries(
      ${_NAME}
      PUBLIC ${CC_LIB_DEPS}
      PRIVATE ${CC_LIB_LINKOPTS})

    # Linker language can be inferred from sources, but in the case of DLLs we
    # don't have any .cc files so it would be ambiguous. We could set it
    # explicitly only in the case of DLLs but, because "CXX" is always the
    # correct linker language for static or for shared libraries, we set it
    # unconditionally.
    set_property(TARGET ${_NAME} PROPERTY LINKER_LANGUAGE "CXX")

    target_include_directories(${_NAME} PUBLIC ${CC_LIB_INCS})
    target_compile_definitions(${_NAME} PUBLIC ${CC_LIB_DEFINES})
    target_compile_options(${_NAME} PRIVATE ${CC_LIB_COPTS})

    # INTERFACE libraries can't have the CXX_STANDARD property set
    set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD 17)
    set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
  else()
    # Generating header-only library
    add_library(${_NAME} INTERFACE)
    target_include_directories(${_NAME} INTERFACE ${CC_LIB_INCS})
    target_link_libraries(${_NAME} INTERFACE ${CC_LIB_DEPS} ${CC_LIB_LINKOPTS})
    target_compile_definitions(${_NAME} INTERFACE ${CC_LIB_DEFINES})
    target_compile_options(${_NAME} INTERFACE ${CC_LIB_COPTS})
  endif()

  if(CC_TARGET_PREFIX)
    add_library(${CC_TARGET_PREFIX}::${CC_LIB_NAME} ALIAS ${_NAME})
  endif()
endfunction()

# cmake-format: off
# cc_binary()
# adapted from absl_cc_test()
#
# Parameters:
# NAME: name of target (see Usage below)
# SRCS: List of source files for the binary
# DEPS: List of other libraries to be linked in to the binary targets
# COPTS: List of private compile options
# DEFINES: List of public definitions
# LINKOPTS: List of link options
#
# Note:
# By default, cc_binary will always create a binary named 
# ${CC_TARGET_PREFIX}_${NAME}.
#
# Usage:
# cc_library(
#   NAME
#     awesome
#   HDRS
#     "a.h"
#   SRCS
#     "a.cc"
#   PUBLIC
# )
#
# cc_binary(
#   NAME
#     awesome_test
#   SRCS
#     "awesome_test.cc"
#   DEPS
#     sv::awesome
#     gmock
#     gtest_main
# )
# cmake-format: on
function(cc_binary)
  cmake_parse_arguments(CC_BIN "" "NAME" "SRCS;COPTS;DEFINES;LINKOPTS;DEPS"
                        ${ARGN})

  if(CC_TARGET_PREFIX)
    set(_NAME "${CC_TARGET_PREFIX}_${CC_BIN_NAME}")
  else()
    set(_NAME ${CC_BIN_NAME})
  endif()

  add_executable(${_NAME} "")
  target_sources(${_NAME} PRIVATE ${CC_BIN_SRCS})

  target_compile_definitions(${_NAME} PUBLIC ${CC_BIN_DEFINES})
  target_compile_options(${_NAME} PRIVATE ${CC_BIN_COPTS})

  target_link_libraries(
    ${_NAME}
    PUBLIC ${CC_BIN_DEPS}
    PRIVATE ${CC_BIN_LINKOPTS})

  set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD 17)
  set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
endfunction()

# cmake-format: off
# cc_test()
# adapted from absl_cc_test()
#
# Parameters:
# NAME: name of target (see Usage below)
# SRCS: List of source files for the binary
# DEPS: List of other libraries to be linked in to the binary targets
# COPTS: List of private compile options
# DEFINES: List of public definitions
# LINKOPTS: List of link options
#
# Note:
# By default, cc_test will always create a binary named ${CC_TARGET_PREFIX}_${NAME}.
# This will also add it to ctest list as ${CC_TARGET_PREFIX}_${NAME}.
#
# Usage:
# cc_library(
#   NAME
#     awesome
#   HDRS
#     "a.h"
#   SRCS
#     "a.cc"
#   PUBLIC
# )
#
# cc_test(
#   NAME
#     awesome_test
#   SRCS
#     "awesome_test.cc"
#   DEPS
#     sv::awesome
#     gmock
# )
# cmake-format: on
function(cc_test)
  cmake_parse_arguments(CC_TEST "" "NAME" "SRCS;COPTS;DEFINES;LINKOPTS;DEPS"
                        ${ARGN})

  if(CC_TARGET_PREFIX)
    set(_NAME "${CC_TARGET_PREFIX}_${CC_TEST_NAME}")
  else()
    set(_NAME ${CC_TEST_NAME})
  endif()

  if(NOT BUILD_TESTING)
    return()
  endif()

  add_executable(${_NAME} "")
  target_sources(${_NAME} PRIVATE ${CC_TEST_SRCS})

  target_compile_definitions(${_NAME} PUBLIC ${CC_TEST_DEFINES})
  target_compile_options(${_NAME} PRIVATE ${CC_TEST_COPTS})
  target_link_libraries(
    ${_NAME}
    PUBLIC ${CC_TEST_DEPS}
    PRIVATE ${CC_TEST_LINKOPTS})

  target_link_libraries(${_NAME} PRIVATE GTest::GTest GTest::Main)

  set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD 17)
  set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

  add_test(NAME ${_NAME} COMMAND ${_NAME})
  set_tests_properties(${_NAME} PROPERTIES FAIL_REGULAR_EXPRESSION ".*FAILED.*")
endfunction()

# cmake-format: off
# cc_bench()
# adapted from absl_cc_test()
#
# Parameters:
# NAME: name of target (see Usage below)
# SRCS: List of source files for the binary
# DEPS: List of other libraries to be linked in to the binary targets
# COPTS: List of private compile options
# DEFINES: List of public definitions
# LINKOPTS: List of link options
#
# Note:
# By default, cc_bench will always create a binary named ${CC_TARGET_PREFIX}_${NAME}.
# This will also add it to ctest list as ${CC_TARGET_PREFIX}_${NAME}.
#
# Usage:
# cc_library(
#   NAME
#     awesome
#   HDRS
#     "a.h"
#   SRCS
#     "a.cc"
#   PUBLIC
# )
#
# cc_bench(
#   NAME
#     awesome_bench
#   SRCS
#     "awesome_bench.cc"
#   DEPS
#     sv::awesome
# )
# cmake-format: on
function(cc_bench)
  if(NOT BUILD_BENCHMARK)
    return()
  endif()

  cmake_parse_arguments(CC_BENCH "" "NAME" "SRCS;COPTS;DEFINES;LINKOPTS;DEPS"
                        ${ARGN})

  if(CC_TARGET_PREFIX)
    set(_NAME "${CC_TARGET_PREFIX}_${CC_BENCH_NAME}")
  else()
    set(_NAME ${CC_BENCH_NAME})
  endif()

  if(NOT benchmark_FOUND)
    message(WARNING "benchmark not found, not building" ${_NAME})
    return()
  endif()

  add_executable(${_NAME} "")
  target_sources(${_NAME} PRIVATE ${CC_BENCH_SRCS})

  target_compile_definitions(${_NAME} PUBLIC ${CC_BENCH_DEFINES})
  target_compile_options(${_NAME} PRIVATE ${CC_BENCH_COPTS})

  target_link_libraries(
    ${_NAME}
    PUBLIC ${CC_BENCH_DEPS}
    PRIVATE ${CC_BENCH_LINKOPTS} benchmark::benchmark benchmark::benchmark_main)

  set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD 17)
  set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
endfunction()

cmake_minimum_required(VERSION 3.17.3)

project(reduction)

set(SRC_FILES
    "reduction.cpp")

set(SYCL_FILES
    "reduction.cpp")

list(APPEND CMAKE_MODULE_PATH
    "${CMAKE_SOURCE_DIR}/cmake")

set(ComputeCpp_DIR CACHE STRING "NOT-FOUND")

if (NOT ComputeCpp_DIR)
    message(FATAL_ERROR
    "SYCL implementation root not provided, please specify "
    "the path to the root of the chosen SYCL implementation using "
    "ComputeCpp_DIR=<path/to/install/root>.")
endif()

find_package(ComputeCpp REQUIRED)

add_executable("${PROJECT_NAME}" "${SRC_FILES}" "${HEADER_FILES}")

target_include_directories("${PROJECT_NAME}" PRIVATE 
                           "${ComputeCpp_INCLUDE_DIRS}"
                           "${CMAKE_SOURCE_DIR}/src")

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17)

target_compile_definitions(${PROJECT_NAME} PUBLIC -DSYCL_LANGUAGE_VERSION=2020)

add_sycl_to_target(TARGET "${PROJECT_NAME}"
                   SOURCES "${SYCL_FILES}")

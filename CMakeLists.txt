cmake_minimum_required(VERSION 3.20)

project(itlab_2024)

include(cmake/configure.cmake)

include_directories(include)

enable_testing()


add_subdirectory(3rdparty)
add_subdirectory(app)
add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(test)

# REPORT
message( STATUS "")
message( STATUS "General configuration for ${PROJECT_NAME}")
message( STATUS "======================================")
message( STATUS "")
message( STATUS "   Configuration: ${CMAKE_BUILD_TYPE}")
message( STATUS "")
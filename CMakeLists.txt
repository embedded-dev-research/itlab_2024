cmake_minimum_required(VERSION 3.20)

set(ProjectName "itlab")

project(${ProjectName})

include_directories(include)

add_subdirectory(3rdparty)
add_subdirectory(src/graph)
add_subdirectory(test)

# REPORT
message( STATUS "")
message( STATUS "General configuration for ${PROJECT_NAME}")
message( STATUS "======================================")
message( STATUS "")
message( STATUS "   Configuration: ${CMAKE_BUILD_TYPE}")
message( STATUS "")
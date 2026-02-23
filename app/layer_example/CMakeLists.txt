set(ARM_DIR "${CMAKE_SOURCE_DIR}/3rdparty/ComputeLibrary")

add_executable(Split SplitLayer.cpp)

include_directories(${ARM_DIR})
include_directories(${ARM_DIR}/include)
target_link_directories(Split PUBLIC ${ARM_DIR}/build)

target_link_libraries(Split arm_compute)

add_dependencies(Split build_compute_library)
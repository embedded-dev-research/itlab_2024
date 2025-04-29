set(ARM_DIR "${CMAKE_SOURCE_DIR}/3rdparty/ComputeLibrary")

add_executable(Reshape ReshapeLayer.cpp)

include_directories(${ARM_DIR})
include_directories(${ARM_DIR}/include)
target_link_directories(Reshape PUBLIC ${ARM_DIR}/build)

target_link_libraries(Reshape arm_compute)

add_dependencies(Reshape build_compute_library)
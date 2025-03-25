set(ARM_DIR "${CMAKE_SOURCE_DIR}/3rdparty/ComputeLibrary")

add_executable(MatMul MatMulLayerLayer.cpp)

include_directories(${ARM_DIR})
include_directories(${ARM_DIR}/include)
link_directories(${ARM_DIR}/build)

target_link_libraries(MatMul arm_compute)

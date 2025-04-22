set(ARM_DIR "${CMAKE_SOURCE_DIR}/3rdparty/ComputeLibrary")

add_executable(MatMul MatMulLayer.cpp)

include_directories(${ARM_DIR})
include_directories(${ARM_DIR}/include)
target_link_directories(MatMul PUBLIC ${ARM_DIR}/build)

target_link_libraries(MatMul arm_compute)

add_dependencies(MatMul build_compute_library)
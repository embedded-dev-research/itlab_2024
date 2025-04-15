set(ARM_DIR "${CMAKE_SOURCE_DIR}/3rdparty/ComputeLibrary")

add_executable(PoolingLayer PoolingLayer.cpp)

include_directories(${ARM_DIR})
include_directories(${ARM_DIR}/include)
target_link_directories(PoolingLayer PUBLIC ${ARM_DIR}/build)

target_link_libraries(PoolingLayer arm_compute)
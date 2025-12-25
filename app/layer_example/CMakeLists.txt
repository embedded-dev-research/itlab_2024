set(ARM_DIR "${CMAKE_SOURCE_DIR}/3rdparty/ComputeLibrary")

add_executable(ElementwiseLayer ElementwiseLayer.cpp)

include_directories(${ARM_DIR})
include_directories(${ARM_DIR}/include)
target_link_directories(ElementwiseLayer PUBLIC ${ARM_DIR}/build)

target_link_libraries(ElementwiseLayer arm_compute)
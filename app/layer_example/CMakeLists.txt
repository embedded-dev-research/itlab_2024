set(ARM_DIR "${CMAKE_SOURCE_DIR}/3rdparty/ComputeLibrary")

add_executable(ElementwiseLayer ElementwiseLayer.cpp)

include_directories(${ARM_DIR})
include_directories(${ARM_DIR}/include)
link_directories(${ARM_DIR}/build)

target_link_libraries(ElementwiseLayer arm_compute)
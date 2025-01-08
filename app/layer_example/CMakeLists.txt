set(ARM_DIR "/3rdparty/ComputeLibrary")

include_directories(${ARM_DIR}/include)
link_directories(${ARM_DIR}/build)

add_executable(ConvolutionLayer ConvolutionLayer.cpp)
add_executable(ElementwiseLayer ElementwiseLayer.cpp)

target_link_libraries(ConvolutionLayer arm_compute)
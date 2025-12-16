set(ARM_DIR "${CMAKE_SOURCE_DIR}/3rdparty/ComputeLibrary")

add_executable(Concat ConcatLayer.cpp)

include_directories(${ARM_DIR})
include_directories(${ARM_DIR}/include)
target_link_directories(Concat PUBLIC ${ARM_DIR}/build)

target_link_libraries(Concat arm_compute)

add_dependencies(Concat build_compute_library)
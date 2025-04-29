set(ARM_DIR "${CMAKE_SOURCE_DIR}/3rdparty/ComputeLibrary")

add_executable(Slice SliceLayer.cpp)

include_directories(${ARM_DIR})
include_directories(${ARM_DIR}/include)
target_link_directories(Slice PUBLIC ${ARM_DIR}/build)

target_link_libraries(Slice arm_compute)

add_dependencies(Slice build_compute_library)
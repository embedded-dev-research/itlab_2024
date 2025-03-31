add_subdirectory(googletest)

set(COMPUTE_LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ComputeLibrary)
set(CMAKE_BUILD_PARALLEL_LEVEL 4)

if (LINUX)
  set(ACL_OS "linux")
elseif(APPLE)
  set(ACL_OS "macos")
else()
  set(ACL_OS "unknown")
endif()

if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
  set(ACL_ARCH "x86")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
  set(ACL_ARCH "armv8.2-a")
else()
  set(ACL_OS "unknown")
endif()

add_custom_target(build_compute_library ALL
    COMMAND scons Werror=1 debug=0 asserts=0 neon=0 opencl=0 os=${ACL_OS} embed_kernels=1 arch=${ACL_ARCH} -j${CMAKE_BUILD_PARALLEL_LEVEL}
    WORKING_DIRECTORY ${COMPUTE_LIBRARY_DIR}
    COMMENT "Building Arm Compute Library with scons"
)
add_custom_target(build_opencv ALL
  COMMAND cmake -S ${CMAKE_CURRENT_SOURCE_DIR}/opencv -B ${CMAKE_CURRENT_SOURCE_DIR}/opencv/build
          -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
          -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
          -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
          -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
          -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  COMMAND cmake --build ${CMAKE_CURRENT_SOURCE_DIR}/opencv/build -j${CMAKE_BUILD_PARALLEL_LEVEL}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/opencv
  COMMENT "Building OpenCV"
)

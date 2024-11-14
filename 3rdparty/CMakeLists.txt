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

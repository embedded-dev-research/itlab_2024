cmake_minimum_required(VERSION 3.5)

project(yolov8-cls-inference)

find_package(OpenCV REQUIRED)

set(PROJECT_SOURCE_FILES
    main.cpp
)

add_executable(${CMAKE_PROJECT_NAME} ${PROJECT_SOURCE_FILES})
target_link_libraries(${CMAKE_PROJECT_NAME} ${OpenCV_LIBS})
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
cmake_minimum_required(VERSION 3.10)
project(ModelParser)

find_package(Protobuf REQUIRED)
set(ONNX_GENERATED_DIR "${CMAKE_SOURCE_DIR}/app/frontend/build/generated")


set(SETUP_SCRIPT "${CMAKE_SOURCE_DIR}/app/frontend/GetModel.sh")

set(GENERATED_FILES
"${ONNX_GENERATED_DIR}/onnx.pb.cc"
"${ONNX_GENERATED_DIR}/onnx.pb.h"
"${ONNX_GENERATED_DIR}/yolo11x.onnx"
)


add_custom_command(
    OUTPUT ${GENERATED_FILES}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${ONNX_GENERATED_DIR}"
    COMMAND ${CMAKE_COMMAND} -E chdir ${ONNX_GENERATED_DIR} bash ${SETUP_SCRIPT} "${ONNX_GENERATED_DIR}"
    WORKING_DIRECTORY "${ONNX_GENERATED_DIR}"
    DEPENDS ${SETUP_SCRIPT}
    COMMENT "Create directory, generate ONNX-files and get YOLO model"
    )
    
add_custom_target(generate_output ALL
DEPENDS ${GENERATED_FILES}
)

set(ONNX_GENERATED_DIR "${CMAKE_SOURCE_DIR}/app/frontend/build/generated")

set_source_files_properties(
  "${ONNX_GENERATED_DIR}/onnx.pb.cc"
  "${ONNX_GENERATED_DIR}/onnx.pb.h"
  "${ONNX_GENERATED_DIR}/yolo11x.onnx"
  PROPERTIES GENERATED TRUE
)

add_executable(ModelParser
  main.cpp
  "${ONNX_GENERATED_DIR}/onnx.pb.cc"
)
add_dependencies(ModelParser generate_output)
target_include_directories(ModelParser PRIVATE
  "${ONNX_GENERATED_DIR}"
  ${Protobuf_INCLUDE_DIRS}
)

target_link_libraries(ModelParser
  ${Protobuf_LIBRARIES}
)

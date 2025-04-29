cmake_minimum_required(VERSION 3.10)
project(ModelParser)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Protobuf REQUIRED)


set(ONNX_GENERATED_DIR "${CMAKE_SOURCE_DIR}/app/frontend/build/generated") 

set(ONNX_PROTO_SRCS
    "${ONNX_GENERATED_DIR}/app/frontend/build/generated/onnx.pb.cc"
)

set(ONNX_GENERATED_HEADER 
    "${ONNX_GENERATED_DIR}/app/frontend/build/generated/onnx.pb.h"
) 

set(SRC_FILES
    main.cpp
    ${ONNX_PROTO_SRCS}
)

set(SETUP_SCRIPT ${CMAKE_SOURCE_DIR}/app/frontend/GetModel.sh)


set(GENERATED_FILES
  ${ONNX_GENERATED_DIR}/onnx.pb.cc
  ${ONNX_GENERATED_DIR}/onnx.pb.h
  ${ONNX_GENERATED_DIR}/yolo11x.onnx
)

add_custom_command(
  OUTPUT ${GENERATED_FILES}
  COMMAND bash ${SETUP_SCRIPT}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  DEPENDS ${SETUP_SCRIPT}
)

add_custom_target(generate_output ALL
  DEPENDS ${GENERATED_FILES}
)


set_source_files_properties(${ONNX_GENERATED_HEADER} PROPERTIES GENERATED TRUE)
set_source_files_properties(${ONNX_PROTO_SRCS} PROPERTIES GENERATED TRUE)
set_source_files_properties(${ONNX_GENERATED_DIR}/yolo11x.onnx PROPERTIES GENERATED TRUE)


add_executable(ModelParser 
  main.cpp
  ${ONNX_GENERATED_DIR}/onnx.pb.cc 
)

target_include_directories(ModelParser PRIVATE
    ${ONNX_GENERATED_DIR} 
    ${Protobuf_INCLUDE_DIRS}
)

target_link_libraries(ModelParser 
    ${Protobuf_LIBRARIES}   
    ${ONNX_GENERATED_DIR}
)

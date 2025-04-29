cmake_minimum_required(VERSION 3.10)
project(ModelParser)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Protobuf REQUIRED)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated)

set(ONNX_GENERATED_DIR "${CMAKE_SOURCE_DIR}/generated") 

set(ONNX_PROTO_SRCS
    "${ONNX_GENERATED_DIR}/onnx.pb.cc"
)

set(ONNX_GENERATED_HEADER 
    "${ONNX_GENERATED_DIR}/onnx.pb.h"
) 

set(SRC_FILES
    main.cpp
    ${ONNX_PROTO_SRCS}
)

add_custom_command(
  OUTPUT 
    ${ONNX_GENERATED_DIR}/onnx.pb.cc
    ${ONNX_GENERATED_DIR}/onnx.pb.h
    ${ONNX_GENERATED_DIR}/yolo11x.onnx
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/GetModel.sh
  WORKING_DIRECTORY ${ONNX_GENERATED_DIR}
  COMMENT "Запуск генерации файлов..."
  VERBATIM
)



set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/generated/ PROPERTIES GENERATED TRUE)
set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/generated/onnx.pb.h PROPERTIES GENERATED TRUE)
set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/generated/onnx.pb.cc PROPERTIES GENERATED TRUE)
set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/generated/yolo11x.onnx PROPERTIES GENERATED TRUE)


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

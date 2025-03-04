cmake_minimum_required(VERSION 3.10)
project(ModelParser)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Protobuf REQUIRED)

set(ONNX_GENERATED_DIR "./generated") 

set(ONNX_PROTO_SRCS
    "./generated/onnx.pb.cc"
)

set(SRC_FILES
    main.cpp
    ${ONNX_PROTO_SRCS}
)

configure_file(${CMAKE_SOURCE_DIR}/generated/yolo11x.onnx
               ${CMAKE_CURRENT_BINARY_DIR}/generated/yolo11x.onnx
               COPYONLY)

add_executable(ModelParser main.cpp ${ONNX_PROTO_SRCS}) 

target_include_directories(ModelParser PRIVATE
    ${ONNX_GENERATED_DIR} 
    ${Protobuf_INCLUDE_DIRS}
)

target_link_libraries(ModelParser ${Protobuf_LIBRARIES})

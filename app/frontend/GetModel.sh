#!/bin/bash

protoc --proto_path=./3rdparty/onnx/onnx --cpp_out=$1 onnx.proto
yolo export model=yolo11x.pt format=onnx save_dir=$1



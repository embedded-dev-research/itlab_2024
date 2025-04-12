#!/bin/bash

protoc --proto_path=../../3rdparty/onnx/onnx --cpp_out=./generated onnx.proto
yolo export model=yolo11x.pt format=onnx save_dir=./generated



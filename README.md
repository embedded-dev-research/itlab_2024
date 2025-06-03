[![Build application](https://github.com/embedded-dev-research/itlab_2024/actions/workflows/main.yml/badge.svg)](https://github.com/embedded-dev-research/itlab_2024/actions/workflows/main.yml)
[![Static analysis](https://github.com/embedded-dev-research/itlab_2024/actions/workflows/static-analysis.yml/badge.svg)](https://github.com/embedded-dev-research/itlab_2024/actions/workflows/static-analysis.yml)
[![CodeQL](https://github.com/embedded-dev-research/itlab_2024/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/embedded-dev-research/itlab_2024/actions/workflows/codeql-analysis.yml)

# itlab_2024


## Build and run ultralytics inference example

1. Clone ultralyrics repository
   ```
   git clone https://github.com/ultralytics/ultralytics
   cd ultralytics
   ```
    
4. Get a YOLO model
   
   We aslo will need a YOLO model in ONNX format.
   
   Create and activate python venv
   ```
   python -m venv <venv dir>
   source <venv dir>/bin/activate
   ```

   Install ultralytics using pip
   ```
   pip install ultralytics
   ```

   Get a YOLO model
   ```
   yolo export model=yolov8s.pt imgsz=480,640 format=onnx
   ```

5. Build and run the example
   ```
   cd examples/YOLOv8-CPP-Inference
   cmake -S . -B build
   cd build
   make
   ./Yolov8CPPInference
   ```
    
   Note that by default the CMake file will try to import the CUDA library to be used with the OpenCVs dnn (cuDNN) GPU Inference.
   If your OpenCV build does not use CUDA/cuDNN you can remove that import call and run the example on CPU.

## How to build ONNX library on Linux(Ubuntu)

1. Install necessary tools:
  ```
  sudo apt-get install -y python3-pip
  sudo apt-get install -y python3-venv
  sudo apt-get install python3-pip python3-dev libprotobuf-dev protobuf-compiler
  ```

2. Open directory where you want to save the [ONNX library](https://github.com/onnx/onnx.git).

3. Open a terminal and execute these commands:
  ```
  cd onnx
  git submodule update --init --recursive
  export CMAKE_ARGS='-DONNX_USE_LITE_PROTO=ON -DONNX_USE_PROTOBUF_SHARED_LIBS=ON'
  ```
*when you use pip, you need to use **venv** to avoid conflict between package managers (apt and pip)*
*in source(onnx) directory execute these commands:*
  ```
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -e . -v
  ```
4. Run to verify it works.:
  ```
   python -c "import onnx"
  ```

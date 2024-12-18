[![Build application](https://github.com/embedded-dev-research/itlab_2024/actions/workflows/main.yml/badge.svg)](https://github.com/embedded-dev-research/itlab_2024/actions/workflows/main.yml)
[![Static analysis](https://github.com/embedded-dev-research/itlab_2024/actions/workflows/static-analysis.yml/badge.svg)](https://github.com/embedded-dev-research/itlab_2024/actions/workflows/static-analysis.yml)
[![CodeQL](https://github.com/embedded-dev-research/itlab_2024/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/embedded-dev-research/itlab_2024/actions/workflows/codeql-analysis.yml)

# itlab_2024


## Build and run ultralytics inference example

1. OpenCV installation
   
    We will need OpenCV library to run the example. You can install it in linux using this [guide](https://docs.opencv.org/4.10.0/d7/d9f/tutorial_linux_install.html)

2. Clone ultralyrics repository
    ```
    git clone https://github.com/ultralytics/ultralytics
    cd ultralytics
    ```
    
4. Get a YOLO model
   
    To run the examples we will also need a YOLO model in ONNX format.
   
    Install ultralyrics using pip (preferably in python venv):
    ```
    pip install ultralytics
    ```

    Activate python venv if you use it
    ```
    source <venv dir>/bin/activate
    ```

    Get a YOLO model
    ```
    yolo export model=yolov8s.pt imgsz=480,640 format=onnx
    ```

5. Build the example
    ```
    cd examples/YOLOv8-CPP-Inference
    cmake -S . -B build
    cd build
    make
    ```
    
    Note that by default the CMake file will try to import the CUDA library to be used with the OpenCVs dnn (cuDNN) GPU Inference.
    If your OpenCV build does not use CUDA/cuDNN you can remove that import call and run the example on CPU.
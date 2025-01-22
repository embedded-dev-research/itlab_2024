[![Build application](https://github.com/embedded-dev-research/itlab_2024/actions/workflows/main.yml/badge.svg)](https://github.com/embedded-dev-research/itlab_2024/actions/workflows/main.yml)
[![Static analysis](https://github.com/embedded-dev-research/itlab_2024/actions/workflows/static-analysis.yml/badge.svg)](https://github.com/embedded-dev-research/itlab_2024/actions/workflows/static-analysis.yml)
[![CodeQL](https://github.com/embedded-dev-research/itlab_2024/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/embedded-dev-research/itlab_2024/actions/workflows/codeql-analysis.yml)

# itlab_2024


## Extract network data using OpenCV and Ultralytics

1. OpenCV installation
   
    If you don't have OpenCV library you can install it for linux using the official guide: [OpenCV installation guide](https://docs.opencv.org/4.10.0/d7/d9f/tutorial_linux_install.html)
    
2. Ultralytics installation
   
    Create and activate python venv
    ```
    python -m venv <venv dir>
    source <venv dir>/bin/activate
    ```

    Install ultralytics using pip:
    ```
    pip install ultralytics
    ```

3. Get a YOLO model
    ```
    yolo export model=yolov8s.pt imgsz=480,640 format=onnx
    ```

4. Build and run the project 
    ```
    cmake -S . -B build
    cd build
    make
    ./extract-network-data
    ```
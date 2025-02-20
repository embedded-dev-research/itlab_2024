[![Build application](https://github.com/embedded-dev-research/itlab_2024/actions/workflows/main.yml/badge.svg)](https://github.com/embedded-dev-research/itlab_2024/actions/workflows/main.yml)
[![Static analysis](https://github.com/embedded-dev-research/itlab_2024/actions/workflows/static-analysis.yml/badge.svg)](https://github.com/embedded-dev-research/itlab_2024/actions/workflows/static-analysis.yml)
[![CodeQL](https://github.com/embedded-dev-research/itlab_2024/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/embedded-dev-research/itlab_2024/actions/workflows/codeql-analysis.yml)

# itlab_2024
## __How to build ONNX library on Linux(Ubuntu)__

1. Install necessary tools:
  
  ```
   `sudo apt install -y python3-pip`
    sudo apt install -y python3-venv`
    sudo apt install python3-pip python3-dev libprotobuf-dev protobuf-compiler
  ```
2. Install package manager *pip*.
3. Open directory where you want to save the [onnx library](https://github.com/onnx/onnx.git).
4. Open a terminal and execute these commands:
```
git clone https://github.com/onnx/onnx.git
cd onnx
git submodule update --init --recursive
export CMAKE_ARGS=-DONNX_USE_LITE_PROTO=ON
```

*when you use pip, you need to use **venv** to avoid conflict between package managers (apt and pip)*

*in source(onnx) directory execute these commands:*

```
python3 -m venv .venv
source .venv/bin/activate
export CMAKE_ARGS="-DONNX_USE_PROTOBUF_SHARED_LIBS=ON"
pip install -e . -v
```

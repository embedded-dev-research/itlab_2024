[![Build application](https://github.com/embedded-dev-research/itlab_2024/actions/workflows/main.yml/badge.svg)](https://github.com/embedded-dev-research/itlab_2024/actions/workflows/main.yml)
[![Static analysis](https://github.com/embedded-dev-research/itlab_2024/actions/workflows/static-analysis.yml/badge.svg)](https://github.com/embedded-dev-research/itlab_2024/actions/workflows/static-analysis.yml)
[![CodeQL](https://github.com/embedded-dev-research/itlab_2024/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/embedded-dev-research/itlab_2024/actions/workflows/codeql-analysis.yml)

# itlab_2024

## __How to build ONNX library on Linux(Ubuntu)__

1. Install [Protocol Buffer](https://github.com/protocolbuffers/protobuf.git):
* In other directory open a terminal and execute these commands:
  ```
  git clone https://github.com/protocolbuffers/protobuf.git
  cd protobuf
  git checkout v21.12
  git submodule update --init --recursive
  mkdir build_source && cd build_source
  cmake ../cmake -Dprotobuf_BUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_INSTALL_SYSCONFDIR=/etc -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
  make -j$(nproc)
  make install
  ```
2. Instal package manager *pip*.
3. Open directory where you want to save the [onnx library](https://github.com/onnx/onnx.git).
4. Open a terminal and execute these commands:
```
git clone https://github.com/onnx/onnx.git
cd onnx
git submodule update --init --recursive
set CMAKE_ARGS=-DONNX_USE_LITE_PROTO=ON
pip3 install -e . -v
```

*when you use pip, you need to use **venv** to avoid conflict between package managers (apt and pip)*

*in source(onnx) directory execute these commands:*
```
python3 -m venv .venv
source .venv/bin/activate
```

>to install pip `sudo apt-get install -y python3-pip`
>
>to install venv `sudo apt-get install -y python3-venv`

[![Build application](https://github.com/embedded-dev-research/itlab_2024/actions/workflows/main.yml/badge.svg)](https://github.com/embedded-dev-research/itlab_2024/actions/workflows/main.yml)
[![Static analysis](https://github.com/embedded-dev-research/itlab_2024/actions/workflows/static-analysis.yml/badge.svg)](https://github.com/embedded-dev-research/itlab_2024/actions/workflows/static-analysis.yml)
[![CodeQL](https://github.com/embedded-dev-research/itlab_2024/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/embedded-dev-research/itlab_2024/actions/workflows/codeql-analysis.yml)

# itlab_2024

## __How to build ONNX library__

1. Instal package manager *pip*.
2. Open directory where you want to save the library.
3. Open a terminal and execute these commands:
```
git clone [https://github.com/onnx/onnx.git](https://github.com/onnx/onnx.git)
cd onnx
git submodule update --init --recursive
pip install -e . -v
```
4. Open the code editor and run the examples.

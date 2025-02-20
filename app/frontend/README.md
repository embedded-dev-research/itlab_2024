## __How to run yolov11x model parser__

1. Create a directory in which the files necessary for the script will be stored
    ```
    mkdir generated
    ```

2. Generate necessary files
    ```
    protoc --proto_path=<relative path to ONNX project>/onnx --cpp_out=<relative path to generated> onnx.proto
    ```

3. Get a network model using Ultralytics CLI
  
-   Give "executable mode" to the script GetModel.sh 
    ```
    chmod +x GetModel.sh
    ```
-   Run script
    ```
    bash GetModel.sh
    ```

4. In CMakeLists.txt and main.cpp change path to generated files and model file

5. Build and run the project 
    ```
    mkdir build && cd build
    cmake ..
    cmake --build build
    ./ModelParser
    ```

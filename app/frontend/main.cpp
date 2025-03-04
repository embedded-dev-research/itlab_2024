#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>


#include "onnx.pb.h"

int main() {
  std::ifstream model_file("generated/yolo11x.onnx", std::ios::binary); 

  
  if (!model_file.is_open()) {
    std::cerr << "Failed to open model" << std::endl;
    return 1;
  }


  onnx::ModelProto model;
  if (!model.ParseFromIstream(&model_file)) {
    std::cerr << "Model parsing error" << std::endl;
    return 1;
  }
  model_file.close();

  std::vector<std::string> Layer;

  for (int i = 0; i < model.graph().node_size(); ++i) {
    const onnx::NodeProto& node = model.graph().node(i);
    Layer.emplace_back(node.op_type());
  }

  for (auto it : Layer) {
    std::cout << it << std::endl;
  }

  return 0;
}

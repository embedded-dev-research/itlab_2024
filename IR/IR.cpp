#include "IR.h"

IR::IR(std::string filename, Model model) : _graph() {
  _fileName = filename;
  _model = model;
  switch (_model) {
    case Model::ONNX:
      _modelParser = new ONNX_ModelParser(_fileName);
      break;

    case Model::OPENCV:
      _modelParser = new OPENCV_ModelParser(_fileName);
      break;

    // case Model::PYTORCH:
    //     _modelParser = new OPENCV_ModelParser(_fileName);
    //     break;
    default:
      break;
  }

  _graph = _modelParser->Parse();
}

const bool IR::operator==(const IR& _other) const {
  std::unordered_map<int, std::string> first_model_map;
  std::unordered_map<int, std::string> second_model_map;

  std::hash<std::string> hashMaker;

  std::vector<std::string> vec = _graph.getLayersTypeVector();
  for (auto it : _graph) {
    first_model_map.insert(it.getId(), to_string(hashMaker(it.vec)));
  }
  for (auto it : _other._graph){
    second_model_map.insert(it.getId(), to_string(hashMaker(it.vec)));
  }

  if (first_model_map == second_model_map){
    return true;
  }
  return false;
}

IR::~IR() {
  delete _modelParser;
  _modelParser = nullptr;
}
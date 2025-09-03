#pragma once

#include <onnx.pb.h>

#include <fstream>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

#include "graph/graph.h"
#include "model_parser/model_parser.h"

class ONNX_ModelParser : public ModelParser {
 private:
  onnx::ModelProto m_model;

  std::unordered_map<std::string, const onnx::TensorProto*> m_weights;

 public:
  ONNX_ModelParser(const std::string& filename);
  Graph Parse() override;

 private:
  bool LoadModel(const std::string& filename);
  bool ParseAttributes(
      const onnx::NodeProto& node,
      std::unordered_map<std::string, std::string>& attributes);
  bool ParseWeightsAndBias(const onnx::NodeProto& node,
                           const std::shared_ptr<AnyLayer>& any_layer);
};

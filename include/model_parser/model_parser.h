#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "graph/graph.h"

class AnyLayer : public Layer {
 public:
  std::string Name;
  std::unordered_map<std::string, std::string> Attributes;
  std::vector<float> Weights;
  std::vector<float> Bias;

  AnyLayer(const std::string& name);
  ~AnyLayer() = default;

  void addNeighbor(Layer* neighbor);
  void removeNeighbor(Layer* neighbor);
};

enum Model { ONNX, PYTORCH, OPENCV };

class ModelParser {
 protected:
  std::string m_filename;

 public:
  virtual ~ModelParser() = default;
  virtual Graph Parse() = 0;
};
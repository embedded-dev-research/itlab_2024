
#ifndef LAYER_H
#define LAYER_H

#include <list>
#include <memory>
#include <string>

#include "tensor/tensor.h"

class Layer {
 public:
  int id;
  std::string name;
  std::list<std::shared_ptr<Layer>> neighbors_;

  Layer() = default;
  Layer(const std::string& name);
  Layer(std::string&& name);

  virtual ~Layer() = default;
  void setID(int id) { this->id = id; }
  int getID() const { return id; }
  virtual void run(const Tensor<double>& input, Tensor<double>& output) {}
  virtual Shape get_output_shape() { return Shape({1, 1, 1, 1}); }

  void addNeighbor(std::shared_ptr<Layer> neighbor);
  void removeNeighbor(std::shared_ptr<Layer> neighbor);
};
#endif
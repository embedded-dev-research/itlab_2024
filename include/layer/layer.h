#ifndef LAYER_H
#define LAYER_H

#include <list>

#include "./tensor/tensor.h"

class Layer {
 protected:
  int id_;

 public:
  Layer() = default;
  virtual ~Layer() = default;
  void setID(int id) { id_ = id; }
  int getID() const { return id_; }
  virtual void run(const Tensor<double>& input, Tensor<double>& output) = 0;
  virtual Shape get_output_shape() = 0;

  void addNeighbor(Layer* neighbor);
  void removeNeighbor(Layer* neighbor);
  std::list<Layer*> neighbors_;
};
#endif
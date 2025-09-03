
#include "layer/layer.h"

Layer::Layer(const std::string& _name) : name(_name) {}

Layer::Layer(std::string&& _name) : name(_name) {}

void Layer::addNeighbor(std::shared_ptr<Layer> neighbor) {
  if (neighbor != nullptr) {
    neighbors_.push_back(neighbor);
  }
}

void Layer::removeNeighbor(std::shared_ptr<Layer> neighbor) {
  neighbors_.remove(neighbor);
}
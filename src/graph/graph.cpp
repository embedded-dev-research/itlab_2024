#include <iostream>
#include <list>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "./graph/graph.h"
#include "./layer/layer.h"
#include "./tensor/tensor.h"

Graph::Graph()
    : inputTensor_({}), outputTensor_(nullptr) {}

bool Graph::bfs_helper(int start, int vert, bool flag,
                       std::vector<int>* v_ord) const {
  std::unordered_map<int, bool> visited;
  std::queue<int> queue;

  queue.push(start);
  visited[start] = true;

  while (!queue.empty()) {
    int current = queue.front();
    queue.pop();

    if (flag && current == vert) {
      return true;
    }

    if (v_ord != nullptr) {
      v_ord->push_back(current);
    }

    if (layers_.count(current) > 0) {
      Layer* current_layer = layers_.at(current);

      for (Layer* neighbor : current_layer->neighbors_) {
        if (visited.find(neighbor->getID()) == visited.end()) {
          visited[neighbor->getID()] = true;
          queue.push(neighbor->getID());
        }
      }
    }
  }

  return false;
}

void Graph::addLayer(Layer& lay) {
  if (layers_.find(lay.getID()) == layers_.end()) {
    layers_[lay.getID()] = &lay;
  }
}

void Graph::getLayers() const {
  if (!this->empty()) {
    for (const auto& layer : layers_) {
      std::cout << layer.first << " ";
    }
    std::cout << '\n';
  }
}

void Graph::addEdge(Layer& layPrev, Layer& layNext) {
  if (layPrev.getID() == layNext.getID()) {
    throw std::invalid_argument("Cannot add edge from a layer to itself.");
  }
  if (layers_.find(layPrev.getID()) == layers_.end()) {
    addLayer(layPrev);
  }
  if (layers_.find(layNext.getID()) == layers_.end()) {
    addLayer(layNext);
  }
  layPrev.addNeighbor(&layNext);  // Используем метод addNeighbor
}

void Graph::removeEdge(Layer& layPrev, Layer& layNext) {
  if (layers_.find(layPrev.getID()) != layers_.end()) {
    layPrev.removeNeighbor(&layNext);
  }
}

void Graph::removeLayer(Layer& lay) {
  int layer_id = lay.getID();

  if (layers_.find(layer_id) == layers_.end()) {
    return;
  }

  for (auto& pair : layers_) {
    pair.second->removeNeighbor(&lay);
  }

  auto it = layers_.find(layer_id);
  if (it != layers_.end()) {
    layers_.erase(it);
  }

  if (start_ == layer_id) {
    start_ = -1;
  }
  if (end_ == layer_id) {
    end_ = -1;
  }
}

int Graph::layerCount() const { return static_cast<int>(layers_.size()); }

int Graph::edgeCount() const {
  int count = 0;
  for (const auto& layer : layers_) {
    count += layer.second->neighbors_.size();
  }
  return count;
}

bool Graph::empty() const { return layers_.empty(); }

bool Graph::hasPath(Layer& layPrev, Layer& layNext) const {
  if (layers_.find(layPrev.getID()) == layers_.end() ||
      layers_.find(layNext.getID()) == layers_.end()) {
    return false;
  }
  return bfs_helper(layPrev.getID(), layNext.getID(), true, nullptr);
}

std::vector<int> Graph::BFS(int start) {
  std::vector<int> v_ord;
  bfs_helper(start, -1, false, &v_ord);
  return v_ord;
}

void Graph::setInput(Layer& lay, Tensor<double>& vec) {
  if (start_ != -1) {
    throw std::runtime_error("Input layer already set.");
  }
  if (!layers_.empty()) {
    addLayer(lay);
  }
  inputTensor_ = vec;
  start_ = lay.getID();
}

void Graph::setOutput(Layer& lay, Tensor<double>& vec) {
  if (end_ != -1) {
    throw std::runtime_error("Output layer already set.");
  }

  if (layers_.find(lay.getID()) == layers_.end()) {
    addLayer(lay);
  }

  end_ = lay.getID();
  outputTensor_ = &vec;
}

void Graph::inference() {
  if (start_ == -1 || end_ == -1) {
    throw std::runtime_error("Input or output layer not set.");
  }

  std::vector<int> traversal = BFS(start_);

  if (traversal.empty() || traversal.back() != end_) {
    throw std::runtime_error("No path from start to end layer found.");
  }

  Tensor<double> current_tensor = inputTensor_;

  for (int layer_id : traversal) {
    if (layers_.find(layer_id) == layers_.end()) {
      throw std::runtime_error("layer_id out of range in traversal.");
    }
    Layer* current_layer = layers_[layer_id];
    Tensor<double> temp_output_tensor(current_layer->get_output_shape());
    current_layer->run(current_tensor, temp_output_tensor);
    current_tensor = temp_output_tensor;

    if (layer_id == end_) {
      if (outputTensor_ == nullptr) {
        throw std::runtime_error("Output tensor pointer is not set.");
      }
      *outputTensor_ = current_tensor;
    }
  }
}

Graph::~Graph() = default;
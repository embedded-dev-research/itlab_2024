#ifndef GRAPH_H
#define GRAPH_H

#include <memory>
#include <unordered_map>
#include <vector>

#include "layer/layer.h"
#include "tensor/tensor.h"

class Graph {
 private:
  std::unordered_map<int, std::shared_ptr<Layer>> layers_;
  Tensor<double> inputTensor_;
  Tensor<double>* outputTensor_;
  int start_ = -1;
  int end_ = -1;
  bool bfs_helper(int start, int vert, bool flag,
                  std::vector<int>* v_ord) const;

 public:
  Graph();

  void addLayer(std::shared_ptr<Layer> lay);
  void addEdge(std::shared_ptr<Layer> layPrev, std::shared_ptr<Layer> layNext);
  void removeEdge(std::shared_ptr<Layer> layPrev,
                  std::shared_ptr<Layer> layNext);
  void removeLayer(std::shared_ptr<Layer> lay);
  int getLayers() const;
  int getEdges() const;
  bool empty() const;
  bool hasPath(std::shared_ptr<Layer> layPrev,
               std::shared_ptr<Layer> layNext) const;
  std::vector<int> BFS(int start);
  void setInput(std::shared_ptr<Layer> lay, Tensor<double>& vec);
  void setOutput(std::shared_ptr<Layer> lay, Tensor<double>& vec);
  void inference();
  ~Graph();
};

#endif
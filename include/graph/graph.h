#ifndef GRAPH_H
#define GRAPH_H

#include <unordered_map>
#include <vector>

#include "./layer/layer.h"
#include "./tensor/tensor.h"

class Graph {
 private:
  std::unordered_map<int, Layer*> layers_;
  Tensor<double> inputTensor_;
  Tensor<double>* outputTensor_;
  int start_ = -1;
  int end_ = -1;
  bool bfs_helper(int start, int vert, bool flag,
                  std::vector<int>* v_ord) const;

 public:
  Graph();

  void addLayer(Layer& lay);
  void addEdge(Layer& layPrev, Layer& layNext);
  void removeEdge(Layer& layPrev, Layer& layNext);
  void removeLayer(Layer& lay);
  int getLayers() const;
  int getEdges() const;
  bool empty() const;
  bool hasPath(Layer& layPrev, Layer& layNext) const;
  std::vector<int> BFS(int start);
  void setInput(Layer& lay, Tensor<double>& vec);
  void setOutput(Layer& lay, Tensor<double>& vec);
  void inference();
  ~Graph();
};

#endif

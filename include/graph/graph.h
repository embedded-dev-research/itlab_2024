#ifndef GRAPH_H
#define GRAPH_H

#include <list>
#include <unordered_map>
#include <vector>

class Vertex {
 private:
  int id_;
  std::list<int> neighbors_;

 public:
  Vertex(int id_);
  void addNeighbor(int neighbor);
  void removeNeighbor(int neighbor);
  void print() const;
  int getId() const;
  const std::list<int>& getNeighbors() const;
};

class Graph {
 private:
  std::unordered_map<int, Vertex*> vertices_;

 public:
  Graph();
  void addVertex(int id_);
  void getVertex() const;
  void addEdge(int u, int v);
  void removeEdge(int u, int v);
  void removeVertex(int id_);
  int vertexCount() const;
  int edgeCount() const;
  bool empty() const;
  void printGraph() const;
  bool bfs_helper(int start, int vert, bool flag, std::vector<int>* v_ord);
  bool hasPath(int u, int v);
  std::vector<int> BFS(int start);
  ~Graph();
};

#endif

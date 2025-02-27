#include "./graph/graph.h"

#include <iostream>
#include <list>
#include <queue>
#include <unordered_map>
#include <vector>

Vertex::Vertex(int id_) : id_(id_) {}

void Vertex::addNeighbor(int neighbor) {
  if (neighbor != id_) {
    neighbors_.push_back(neighbor);
  }
}

void Vertex::removeNeighbor(int neighbor) { neighbors_.remove(neighbor); }

void Vertex::print() const {
  std::cout << id_ << ": ";
  for (const int& neighbor : neighbors_) {
    std::cout << neighbor << " ";
  }
  std::cout << '\n';
}

int Vertex::getId() const { return id_; }

const std::list<int>& Vertex::getNeighbors() const { return neighbors_; }

Graph::Graph() = default;

void Graph::addVertex(int id_) {
  if (vertices_.find(id_) == vertices_.end()) {
    vertices_[id_] = new Vertex(id_);
  }
}

void Graph::getVertex() const {
  if (!this->empty()) {
    for (const auto& vertice : vertices_) {
      std::cout << vertice.first << " ";
    }
    std::cout << '\n';
  }
}

void Graph::addEdge(int u, int v) {
  if (vertices_.find(u) == vertices_.end()) {
    addVertex(u);
  }
  if (vertices_.find(v) == vertices_.end()) {
    addVertex(v);
  }
  vertices_[u]->addNeighbor(v);
}

void Graph::removeEdge(int u, int v) {
  if (vertices_.find(u) != vertices_.end()) {
    vertices_[u]->removeNeighbor(v);
  }
}

void Graph::removeVertex(int id_) {
  for (auto& pair : vertices_) {
    pair.second->removeNeighbor(id_);
  }
  auto it = vertices_.find(id_);
  if (it != vertices_.end()) {
    delete it->second;
    vertices_.erase(it);
  }
}

int Graph::vertexCount() const {
  int count = 0;
  for (const auto& vertice : vertices_) {
    count++;
  }
  return count;
}

int Graph::edgeCount() const {
  int count = 0;
  for (const auto& vertice : vertices_) {
    count += (vertice.second->getNeighbors()).size();
  }
  return count;
}

bool Graph::empty() const { return vertices_.empty(); }

void Graph::printGraph() const {
  for (const auto& pair : vertices_) {
    pair.second->print();
  }
}

bool Graph::bfs_helper(int start, int vert, bool flag,
                       std::vector<int>* v_ord) {
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

    if (vertices_.find(current) != vertices_.end()) {
      for (const int& neighbor : vertices_[current]->getNeighbors()) {
        if (!visited[neighbor]) {
          visited[neighbor] = true;
          queue.push(neighbor);
        }
      }
    }
  }
  return false;
}

bool Graph::hasPath(int u, int v) {
  if (vertices_.find(u) == vertices_.end() ||
      vertices_.find(v) == vertices_.end()) {
    return false;
  }
  return bfs_helper(u, v, true, nullptr);
}

std::vector<int> Graph::BFS(int start) {
  std::vector<int> v_ord;
  bfs_helper(start, -1, false, &v_ord);
  return v_ord;
}

Graph::~Graph() {
  for (auto& pair : vertices_) {
    delete pair.second;
  }
}
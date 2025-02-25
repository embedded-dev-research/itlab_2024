#include "./graph/graph.h"

Vertex::Vertex(int id) : id(id) {}

void Vertex::addNeighbor(int neighbor) {
    if (neighbor != id) {
        neighbors.push_back(neighbor);
    }
}

void Vertex::removeNeighbor(int neighbor) {
    neighbors.remove(neighbor);
}

void Vertex::print() const {
    std::cout << id << ": ";
    for (const int& neighbor : neighbors) {
        std::cout << neighbor << " ";
    }
    std::cout << std::endl;
}

int Vertex::getId() const {
    return id;
}

const std::list<int>& Vertex::getNeighbors() const {
    return neighbors;
}

Graph::Graph() {}

void Graph::addVertex(int id) {
    if (vertices.find(id) == vertices.end()) {
        vertices[id] = new Vertex(id);
    }
}

void Graph::getVertex() const {
    if (!this->empty()) {
        for (auto it = vertices.begin(); it != vertices.end(); it++)
            std::cout << it->first << " ";
        std::cout << std::endl;
    }
}

void Graph::addEdge(int u, int v) {
    if (vertices.find(u) == vertices.end()) { addVertex(u); }
    if (vertices.find(v) == vertices.end()) { addVertex(v); }
    vertices[u]->addNeighbor(v);
}

void Graph::removeEdge(int u, int v) {
    if (vertices.find(u) != vertices.end()) {
        vertices[u]->removeNeighbor(v);
    }
}

void Graph::removeVertex(int id) {
    for (auto& pair : vertices) { pair.second->removeNeighbor(id); }
    auto it = vertices.find(id);
    if (it != vertices.end()) {
        delete it->second;
        vertices.erase(it);
    }
}

int Graph::vertexCount() const {
    int count = 0;
    for (auto it = vertices.begin(); it != vertices.end(); it++)
        count++;
    return count;
}

int Graph::edgeCount() const {
    int count = 0;
    for (auto it = vertices.begin(); it != vertices.end(); it++) {
        count += (it->second->getNeighbors()).size();
    }
    return count;
}


bool Graph::empty() const {
    return vertices.empty();
}

void Graph::printGraph() const {
    for (const auto& pair : vertices) {
        pair.second->print();
    }
}

bool Graph::hasPath(int u, int v) {
  if (vertices.find(u) == vertices.end() ||
      vertices.find(v) == vertices.end()) {
    return false;
  }
  return bfs_helper(u, v, true, nullptr);
}

std::vector<int> Graph::BFS(int start) {
  std::vector<int> v_ord;
  bfs_helper(start, -1, false, &v_ord);
  return v_ord;
}

bool Graph::bfs_helper(int start, int vert, bool flag, std::vector<int>* v_ord) {
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

    if (vertices.find(current) != vertices.end()) {
      for (const int& neighbor : vertices[current]->getNeighbors()) {
        if (!visited[neighbor]) {
          visited[neighbor] = true;
          queue.push(neighbor);
        }
      }
    }
  }
  return false;
}

Graph::~Graph() {
    for (auto& pair : vertices) {
        delete pair.second;
    }
}
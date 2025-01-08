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
    if (vertices.find(u) != vertices.end() && vertices.find(v) != vertices.end()) {
        std::unordered_map <int, bool> visited;
        std::queue<int> queue;
        queue.push(u);

        while (!queue.empty()) {
            int current = queue.front();
            queue.pop();
            if (current == v) { return true; }
            visited[current] = true;

            for (const int& neighbor : vertices[current]->getNeighbors())
                if (!visited[neighbor])
                    queue.push(neighbor);

        }
        return false;
    }
}

std::vector<int> Graph::BFS(int start) {
    std::vector<int> traversal_order;
    std::unordered_map<int, bool> visited;
    std::queue<int> queue;

    visited[start] = true;
    queue.push(start);

    while (!queue.empty()) {
        int current = queue.front();
        queue.pop();
        traversal_order.push_back(current);

        if (vertices.find(current) != vertices.end()) { 
            for (const int& neighbor : vertices[current]->getNeighbors()) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    queue.push(neighbor);
                }
            }
        }
    }
    //for (int i : traversal_order)
    //    std::cout << i << " ";
    return traversal_order;
}

Graph::~Graph() {
    for (auto& pair : vertices) {
        delete pair.second;
    }
}
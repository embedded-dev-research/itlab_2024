#ifndef GRAPH_H
#define GRAPH_H

#include <iostream>
#include <unordered_map>
#include <list>
#include <queue>

class Vertex {
private:
    int id;
    std::list<int> neighbors;

public:
    Vertex(int id);
    void addNeighbor(int neighbor);
    void removeNeighbor(int neighbor);
    void print() const;
    int getId() const;
    const std::list<int>& getNeighbors() const;
};



class Graph {
private:
    std::unordered_map<int, Vertex*> vertices;

public:

    Graph();

    void addVertex(int id);
    void getVertex() const;
    void addEdge(int u, int v);
    void removeEdge(int u, int v);
    void removeVertex(int id);
    int vertexCount() const;
    int edgeCount() const;
    bool empty() const;
    void printGraph() const;
    bool hasPath(int u, int v);
    std::vector<int> BFS(int start);
    ~Graph();
};

#endif

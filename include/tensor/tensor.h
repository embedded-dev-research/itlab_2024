#ifndef TENSOR_H
#define TENSOR_H

#include <cstddef>
#include <vector>

struct Shape {
  std::vector<size_t> dimensions;
  size_t total_elements;

  Shape(std::vector<size_t> dims);

  size_t get_rank() const;
};

enum Layout : std::uint8_t { kNchw, kNhwc, kNd };

template <typename T>
class Tensor {
 public:
  Shape shape;
  Layout layout;
  std::vector<T> data;

  Tensor(const Shape &sh, Layout l = Layout::kNd);
  Tensor(std::vector<size_t> dims, Layout l = Layout::kNd);

  size_t get_linear_index(const std::vector<size_t> &indices) const;

  T &at(const std::vector<size_t> &indices);
  const T &at(const std::vector<size_t> &indices) const;
};

#endif
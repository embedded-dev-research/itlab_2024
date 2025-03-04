#include "./tensor/tensor.h"

Shape::Shape(std::vector<size_t> dims) : dimensions(std::move(dims)) {
  total_elements = std::accumulate(dimensions.begin(), dimensions.end(), 1ULL,
                                   std::multiplies<>());
}

size_t Shape::get_rank() const { return dimensions.size(); }

template <typename T>
Tensor<T>::Tensor(const Shape &sh, Layout l)
    : shape(sh), layout(l), data(sh.total_elements) {
  if (sh.get_rank() == 4 && l == Layout::ND) {
    std::cerr << "4D Tensor created with ND layout." << std::endl;
  }
}

template <typename T>
Tensor<T>::Tensor(std::vector<size_t> dims, Layout l)
    : Tensor(Shape(std::move(dims)), l) {}


template <typename T>
size_t Tensor<T>::get_linear_index(const std::vector<size_t> &indices) const {
  if (indices.size() != shape.get_rank()) {
    throw std::runtime_error("Incorrect number of indices provided.");
  }
  for (size_t i = 0; i < indices.size(); ++i) {
    if (indices[i] >= shape.dimensions[i]) {
      throw std::out_of_range("Index out of range for dimension");
    }
  }

  size_t linear_index = 0;
  size_t stride = 1;

  if (shape.get_rank() == 4) {
    if (layout == Layout::NCHW) {
      linear_index = indices[0] * (shape.dimensions[1] * shape.dimensions[2] *
                                   shape.dimensions[3]) +
                     indices[1] * (shape.dimensions[2] * shape.dimensions[3]) +
                     indices[2] * shape.dimensions[3] + indices[3];
    } else if (layout == Layout::NHWC) {
      linear_index = indices[0] * (shape.dimensions[1] * shape.dimensions[2] *
                                   shape.dimensions[3]) +
                     indices[1] * (shape.dimensions[2] * shape.dimensions[3]) +
                     indices[2] * shape.dimensions[3] + indices[3];
    } else {
      linear_index = indices[0] * (shape.dimensions[1] * shape.dimensions[2] *
                                   shape.dimensions[3]) +
                     indices[1] * (shape.dimensions[2] * shape.dimensions[3]) +
                     indices[2] * shape.dimensions[3] + indices[3];
    }
  } else {
    std::vector<size_t> reversed_dims = shape.dimensions;
    std::reverse(reversed_dims.begin(), reversed_dims.end());
    for (int i = static_cast<int>(reversed_dims.size()) - 1; i >= 0; --i) {
      linear_index += indices[i] * stride;
      stride *= reversed_dims[i];
    }
  }

  return linear_index;
}

template <typename T>
T &Tensor<T>::at(const std::vector<size_t> &indices) {
  return data[get_linear_index(indices)];
}

template <typename T>
const T &Tensor<T>::at(const std::vector<size_t> &indices) const {
  return data[get_linear_index(indices)];
}
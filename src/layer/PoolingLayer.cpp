#ifndef ACL_POOLING_LAYER_MOCK_H
#define ACL_POOLING_LAYER_MOCK_H

#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "./layer/layer.h"
#include "./tensor/tensor.h"

enum class PoolingType { MAX, AVG, L2 };

struct PoolingLayerInfo {
  PoolingType pool_type{PoolingType::MAX};
  int pool_size_x{2};
  int pool_size_y{2};
  int stride_x{1};
  int stride_y{1};
  int pad_x{0};
  int pad_y{0};
  bool exclude_padding{true};
};

class PoolingLayerMock : public Layer {
 private:
  PoolingLayerInfo pool_info_;
  Shape input_shape_;
  Shape output_shape_;
  bool configured_ = false;

 public:
  PoolingLayerMock(int id, const PoolingLayerInfo& info) : pool_info_(info) {
    setID(id);
  }

  void configure(const Shape& input_shape, Shape& output_shape_ref) {
    if (input_shape.get_rank() != 4) {
      throw std::runtime_error("PoolingMock: Input must be a 4D tensor (e.g., NCHW or NHWC) for this mock.");
    }
    size_t H_in_idx = input_shape.get_rank() - 2;
    size_t W_in_idx = input_shape.get_rank() - 1;

    size_t H_in = input_shape.dimensions[H_in_idx];
    size_t W_in = input_shape.dimensions[W_in_idx];

    size_t H_out = ((H_in + 2 * pool_info_.pad_y - pool_info_.pool_size_y) / pool_info_.stride_y) + 1;
    size_t W_out = ((W_in + 2 * pool_info_.pad_x - pool_info_.pool_size_x) / pool_info_.stride_x) + 1;

    input_shape_ = input_shape;
    output_shape_ = input_shape;

    output_shape_.dimensions[H_in_idx] = H_out;
    output_shape_.dimensions[W_in_idx] = W_out;

    output_shape_ = Shape(output_shape_.dimensions);

    output_shape_ref = output_shape_;
    configured_ = true;
  }

  void exec(const Tensor<double>& input, Tensor<double>& output) override {
    if (!configured_) {
      throw std::runtime_error("PoolingLayerMock: Layer not configured before exec.");
    }
    if (input.shape.dimensions != input_shape_.dimensions) {
      throw std::runtime_error("PoolingLayerMock: Input shape mismatch in exec.");
    }
    if (output.shape.dimensions != output_shape_.dimensions || output.shape.total_elements != output_shape_.total_elements) {
      throw std::runtime_error("PoolingLayerMock: Output shape mismatch in exec.");
    }

    double fill_value = 0.0;
    switch (pool_info_.pool_type) {
      case PoolingType::MAX:
        fill_value = 1.0;
        break;
      case PoolingType::AVG:
        fill_value = 0.5;
        break;
      case PoolingType::L2:
        fill_value = 0.7;
        break;
    }
    std::fill(output.data.begin(), output.data.end(), static_cast<double>(getID()) + fill_value + 0.2);
  }

  Shape get_output_shape() override {
    if (!configured_) {
      throw std::runtime_error("PoolingLayerMock: Layer not configured to get output shape.");
    }
    return output_shape_;
  }

  std::string get_type_name() const override { return "PoolingLayerMock"; }
};

#endif
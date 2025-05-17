#ifndef ACL_CONCATENATE_LAYER_H
#define ACL_CONCATENATE_LAYER_H

#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "include/layer/layer.h"

class ConcatenateLayer : public Layer {
 private:
  std::vector<TensorShape> input_shapes_config_;
  TensorShape output_shape_;
  unsigned int concatenation_axis_;
  bool configured_ = false;

 public:
  ConcatenateLayer(int id) { setID(id); }

  void configure(const std::vector<TensorShape>& inputs_shapes, unsigned int axis, TensorShape& output_shape_ref) {
    if (inputs_shapes.empty()) {
      throw std::runtime_error("Concat: Input shapes list cannot be empty.");
    }

    input_shapes_config_ = inputs_shapes;
    concatenation_axis_ = axis;
    output_shape_ = output_shape_ref;
    configured_ = true;
  }

  void exec(std::vector<const ITensor*>& input, Tensor& output) {
    if (!configured_) {
      throw std::runtime_error("ConcatenateLayer: Layer not configured.");
    }
    if (input.size() != input_shapes_config_.size()) {
      throw std::runtime_error("ConcatenateLayer: different sizes of vectors.");
    }

    output.allocator()->init(TensorInfo(output_shape_, 1, DataType::F32));

    NEConcatenateLayer concat;
    concat.configure(input, &output, concatenation_axis_);
    output.allocator()->allocate();

    concat.run();
  }

  std::string get_type_name() const override {
    return "ConcatenateLayer";
  }
};

#endif
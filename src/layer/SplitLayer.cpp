#ifndef ACL_SPLIT_LAYER_H
#define ACL_SPLIT_LAYER_H

#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "include/layer/layer.h"

class SplitLayer : public Layer {
 private:
  TensorShape input_shape_config_;
  std::vector<TensorShape> output_shapes_computed_;
  unsigned int split_axis_;

  bool configured_ = false;

 public:
  SplitLayer(int id) { setID(id); }

  void configure(const TensorShape& input_shape, unsigned int axis, TensorShape& first_output_shape_ref) {
    input_shape_config_ = input_shape;
    split_axis_ = axis;

    configured_ = true;
  }

  void exec(Tensor& input, std::vector<ITensor*>& outputs) {
    if (!configured_) {
      throw std::runtime_error("SplitLayer: Layer not configured.");
    }
    
    input.allocator()->init(TensorInfo(input_shape_config_, 1, DataType::F32));
    input.allocator()->allocate();

    NESplit split;
    split.configure(&input, outputs, split_axis_);

    split.run();
  }

  std::string get_type_name() const override { return "SplitLayer"; }
};

#endif
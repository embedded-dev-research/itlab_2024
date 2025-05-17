#ifndef ACL_SLICE_LAYER_H
#define ACL_SLICE_LAYER_H

#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "include/layer/layer.h"

class SliceLayer : public Layer {
 private:
  TensorShape input_shape_config_;
  TensorShape output_shape_;
  Coordinates slice_starts_;
  Coordinates slice_ends_;
  bool configured_ = false;

 public:
  SliceLayer(int id) { setID(id); }

  void configure(const TensorShape& input_shape, Coordinates starts, Coordinates ends, TensorShape& output_shape_ref) {
    input_shape_config_ = input_shape;
    slice_starts_ = starts;
    slice_ends_ = ends;
    output_shape_ = output_shape_ref;

    configured_ = true;
  }

  void exec(Tensor& input, Tensor& output) override {
    if (!configured_) {
      throw std::runtime_error("SliceLayer: Layer not configured.");
    }

    input.allocator()->init(TensorInfo(input_shape_config_, 1, DataType::F32));
    output.allocator()->init(TensorInfo(output_shape_, 1, DataType::F32));

    input.allocator()->allocate();
    output.allocator()->allocate();

    NESlice slice;
    slice.configure(&input, &output, slice_starts_, slice_ends_);

    slice.run();
  }


  std::string get_type_name() const override { return "SliceLayer"; }
};

#endif
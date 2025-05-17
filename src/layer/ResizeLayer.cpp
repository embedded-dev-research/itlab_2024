#ifndef ACL_RESIZE_LAYER_H
#define ACL_RESIZE_LAYER_H

#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "include/layer/layer.h"

using namespace arm_compute;
using namespace utils;

class ResizeLayer : public Layer {
private:
  TensorShape input_shape_;
  TensorShape output_shape_;
  bool configured_ = false;

public:
  ResizeLayer(int id) { setID(id); }

  void configure(TensorShape& input_shape, TensorShape& output_shape) {
    input_shape_ = input_shape;
    output_shape_ = output_shape;

    configured_ = true;
  }

  void exec(Tensor& input, Tensor& output) override {
    if (!configured_) {
      throw std::runtime_error(
          "ResizeLayer: Layer not configured before exec.");
    }

    input.allocator()->init(TensorInfo(input_shape_, 1, DataType::F32));
    output.allocator()->init(TensorInfo(output_shape_, 1, DataType::F32));

    input.allocator()->allocate();
    output.allocator()->allocate();

    NEScale resize;
    resize.configure(&input, &output,
                     ScaleKernelInfo{
                         InterpolationPolicy::NEAREST_NEIGHBOR,
                         BorderMode::REPLICATE,
                         PixelValue(),
                         SamplingPolicy::CENTER,
                     });

    resize.run();
  }

  std::string get_type_name() const override { return "ResizeLayer"; }
};

#endif
#ifndef ACL_POOLING_LAYER_H
#define ACL_POOLING_LAYER_H

#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "include/layer/layer.h"

class PoolingLayer : public Layer {
private:
    PoolingLayerInfo pool_info_;
    TensorShape input_shape_;
    TensorShape output_shape_;
    bool configured_ = false;

public:
    PoolingLayer(int id) {
        setID(id);
    }

    void configure(TensorShape& input_shape, PoolingLayerInfo pli = PoolingLayerInfo(PoolingType::MAX,  DataLayout::NHWC), TensorShape& output_shape_ref) {
        if (input_shape.num_dimensions() < 2) {
            throw std::runtime_error("PoolingLayer: Input must be at least 2D");
        }
        pool_info_ = pli;
        input_shape_ = input_shape;
        output_shape_ = input_shape;

        output_shape_ = output_shape_ref;

        configured_ = true;
  }

  void exec(Tensor& input, Tensor& output) override {
    if (!configured_) {
      throw std::runtime_error("PoolingLayer: Layer not configured before exec.");
    }

    input.allocator()->init(TensorInfo(input_shape_, 1, DataType::F32));
    output.allocator()->init(TensorInfo(output_shape_, 1, DataType::F32));

    input.allocator()->allocate();
    output.allocator()->allocate();

    NEPoolingLayer pool;
    pool.configure(&input, &output, pool_info_);
    pool.run();
  }

  std::string get_type_name() const override { return "PoolingLayer"; }
};

#endif
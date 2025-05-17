#ifndef ACL_SOFTMAX_LAYER_H
#define ACL_SOFTMAX_LAYER_H

#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "include/layer/layer.h"

using namespace arm_compute;
using namespace utils;

class SoftmaxLayer : public Layer {
private:
    TensorShape input_shape_;
    TensorShape output_shape_;
    bool configured_ = false;

public:
    SoftmaxLayer(int id) { setID(id); }

    void configure(TensorShape& input_shape, TensorShape& output_shape_ref) {
        input_shape_ = input_shape;
        output_shape_ = output_shape_ref;

        configured_ = true;
    }

    void exec(Tensor& input, Tensor& output) override {
        if (!configured_) {
            throw std::runtime_error(
          "SoftmaxLayer: Layer not configured before exec.");
        }

        input.allocator()->init(TensorInfo(input_shape_, 1, DataType::F32));
        output.allocator()->init(TensorInfo(output_shape_, 1, DataType::F32));

        input.allocator()->allocate();
        output.allocator()->allocate();

        NESoftmaxLayer m;
        m.configure(&input, &output);
        m.run();
    }

    std::string get_type_name() const override { return "SoftmaxLayer"; }
};

#endif
#ifndef ACL_RESHAPE_LAYER_H
#define ACL_RESHAPE_LAYER_H

#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "include/layer/layer.h"

class ReshapeLayer : public Layer {
private:
    TensorShape input_shape_config_;
    TensorShape target_output_shape_config_;
    bool configured_ = false;

public:
    ReshapeLayer(int id) { setID(id); }

    void configure(const TensorShape& input_shape, const TensorShape& target_output_shape, TensorShape& output_shape_ref) {
        input_shape_config_ = input_shape;
        target_output_shape_config_ = target_output_shape;

        configured_ = true;
    }

    void exec(Tensor& input, Tensor& output) override {
        if (!configured_) {
            throw std::runtime_error("ReshapeLayer: Layer not configured.");
        }

        input.allocator()->init(TensorInfo(input_shape_config_, 1, DataType::F32));
        output.allocator()->init(TensorInfo(target_output_shape_config_, 1, DataType::F32));

        input.allocator()->allocate();
        output.allocator()->allocate();

        NEReshapeLayer reshape;
        reshape.configure(&input, &output);

        reshape.run();
    }

  std::string get_type_name() const override { return "ReshapeLayer"; }
};

#endif
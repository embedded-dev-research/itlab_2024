#ifndef ACL_MATMUL_LAYER_H
#define ACL_MATMUL_LAYER_H

#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "include/layer/layer.h"

using namespace arm_compute;
using namespace utils;

class MatMulLayer : public Layer {
private:
    MatMulInfo matmul_info_;
    TensorShape input_x_shape_;
    TensorShape input_y_shape_;
    TensorShape output_shape_;
    bool configured_ = false;

public:
    MatMulLayer(int id, const MatMulInfo& info = MatMulInfo()) : matmul_info_(info) {
        setID(id);
    }

    void configure(TensorShape& input_x_shape, TensorShape& input_y_shape, TensorShape& output_shape_ref) {
        input_x_shape_ = input_x_shape;
        input_y_shape_ = input_y_shape;
        output_shape_ = output_shape_ref;

        configured_ = true;
    }

  void exec(Tensor& input_x, Tensor& input_y, Tensor& output) override {
    if (!configured_) {
      throw std::runtime_error("MatMulLayer: Layer not configured before exec.");
    }

    input_x.allocator()->init(TensorInfo(input_x_shape_, 1, DataType::F32));
    input_y.allocator()->init(TensorInfo(input_y_shape_, 1, DataType::F32));
    output.allocator()->init(TensorInfo(output_shape_, 1, DataType::F32));

    input_x.allocator()->allocate();
    input_y.allocator()->allocate();
    output.allocator()->allocate();

    NEMatMul m;
    m.configure(&input_x, &input_y, &output, matmul_info_, CpuMatMulSettings(), ActivationLayerInfo());
    m.run();
  }

  std::string get_type_name() const override { return "MatMulLayer"; }
};

#endif
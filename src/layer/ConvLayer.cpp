#ifndef ACL_CONVOLUTION_LAYER_SIMPLIFIED_H
#define ACL_CONVOLUTION_LAYER_SIMPLIFIED_H

#include <numeric> 
#include <stdexcept>
#include <string>
#include <vector>

#include "include/layer/layer.h"

class ConvolutionLayer : public Layer {
private:

  TensorShape input_shape_;
  TensorShape weights_shape_;
  TensorShape biases_shape_;
  TensorShape output_shape_;
  Tensor* biase_t;
  Tensor* weight_t;
  PadStrideInfo psi;

  bool configured_ = false;

public:
  ConvolutionLayer(int id) { setID(id); }

  void configure(
      const TensorShape& input_s,    
      const TensorShape& weights_s,
      Tensor& weights_t,
      const TensorShape& biases_s,
      Tensor& biases_t,
      TensorShape& output_s_ref,
      const PadStrideInfo& info
  ) {

    input_shape_ = input_s;
    weights_shape_ = weights_s;
    biases_shape_ = biases_s;
    psi = info;
    output_shape_ = output_s_ref;

    NECopy copyb, copyw;
    copyb.configure(biase_t, &biases_t);
    copyb.run();
    copyw.configure(weight_t, &weights_t);
    copyw.run();

    weight_t->allocator()->init(TensorInfo(weights_shape_, 1, DataType::F32));
    biase_t->allocator()->init(TensorInfo(biases_shape_, 1, DataType::F32));

    weight_t->allocator()->allocate();
    biase_t->allocator()->allocate();

    configured_ = true;
  }

    void exec(Tensor& input, Tensor& output) override {
        if (!configured_) {
            throw std::runtime_error("ConvolutionLayer: Layer not configured.");
        }

        input.allocator()->init(TensorInfo(input_shape_, 1, DataType::F32));
        output.allocator()->init(TensorInfo(output_shape_, 1, DataType::F32));

        input.allocator()->allocate();
        output.allocator()->allocate();

        NEConvolutionLayer conv;
        conv.configure(&input, weight_t, biase_t, &output, psi);
        conv.run();
    }

    std::string get_type_name() const override {
        return "ConvolutionLayer";
    }
};

#endif 
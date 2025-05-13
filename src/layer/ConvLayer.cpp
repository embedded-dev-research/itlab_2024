#ifndef ACL_CONVOLUTION_LAYER_MOCK_SIMPLIFIED_H
#define ACL_CONVOLUTION_LAYER_MOCK_SIMPLIFIED_H

#include <numeric> 
#include <stdexcept>
#include <string>
#include <vector>

#include "./layer/layer.h"
#include "./tensor/tensor.h" 

struct ConvPadStrideInfo {
  unsigned int stride_x{1};
  unsigned int stride_y{1};
  unsigned int pad_x{0};
  unsigned int pad_y{0}; 

  ConvPadStrideInfo(unsigned int sx = 1, unsigned int sy = 1, unsigned int px = 0, unsigned int py = 0): stride_x(sx), stride_y(sy), pad_x(px), pad_y(py) {}
};

class ConvolutionLayerMock : public Layer {
 private:
  ConvPadStrideInfo conv_info_;

  Shape input_shape_config_;
  Shape weights_shape_config_;
  Shape biases_shape_config_;
  Shape output_shape_computed_;

  bool has_biases_ = false;
  bool configured_ = false;

 public:
  ConvolutionLayerMock(int id) { setID(id); }

  void configure(
      const Shape& input_s,    
      const Shape& weights_s,
      const Shape* biases_s,
      Shape& output_s_ref,
      const ConvPadStrideInfo& info
  ) {

    input_shape_config_ = input_s;
    weights_shape_config_ = weights_s;
    conv_info_ = info;

    if (biases_s) {
      has_biases_ = true;
      biases_shape_config_ = *biases_s;
    } else {
      has_biases_ = false;
      biases_shape_config_ = Shape();
    }

    if (input_s.get_rank() < 3)
      throw std::runtime_error("ConvMockSimp: Input rank must be at least 3 (W, H, C).");
    if (weights_s.get_rank() != 4)
      throw std::runtime_error("ConvMockSimp: Weights rank must be 4 (KW, KH, IC, OC).");
    size_t W_in = input_s.dimensions[0];
    size_t H_in = input_s.dimensions[1];
    size_t C_in = input_s.dimensions[2];
    size_t KW = weights_s.dimensions[0];  
    size_t KH = weights_s.dimensions[1];    
    size_t IC_w = weights_s.dimensions[2]; 
    size_t OC_w = weights_s.dimensions[3];

    if (C_in != IC_w) {
      throw std::runtime_error("ConvMockSimp: Input channels mismatch with weights input channels ().");
    }

    if (has_biases_) {
      if (biases_shape_config_.get_rank() != 1 || biases_shape_config_.dimensions[0] != OC_w) {
        throw std::runtime_error("ConvMockSimp: Biases must be 1D and size must match output channels");
      }
    }

    size_t effective_kernel_w = KW;
    size_t effective_kernel_h = KH;

    if (H_in + 2 * conv_info_.pad_y < effective_kernel_h ||
        W_in + 2 * conv_info_.pad_x < effective_kernel_w) {
      throw std::runtime_error("ConvMockSimp: Kernel size is larger than padded input dimensions.");
    }

    size_t W_out = ((W_in + 2 * conv_info_.pad_x - effective_kernel_w) / conv_info_.stride_x) + 1;
    size_t H_out = ((H_in + 2 * conv_info_.pad_y - effective_kernel_h) / conv_info_.stride_y) + 1;

    std::vector<size_t> output_dims = {W_out, H_out, OC_w};
    if (input_s.get_rank() > 3) {
      output_dims.push_back(input_s.dimensions[3]); 
    }

    output_shape_computed_ = Shape(output_dims);
    output_s_ref = output_shape_computed_; 
    configured_ = true;
  }

  void exec(const Tensor<double>& input, Tensor<double>& output) override {
    if (!configured_) {
      throw std::runtime_error("ConvolutionLayerMock: Layer not configured.");
    }
    if (input.shape.dimensions != input_shape_config_.dimensions) {
      throw std::runtime_error("ConvolutionLayerMock: Input shape mismatch with configured shape.");
    }
    if (output.shape.dimensions != output_shape_computed_.dimensions) {
      throw std::runtime_error("ConvolutionLayerMock: Output shape mismatch with computed shape.");
    }

    double fill_value = static_cast<double>(getID()) + 0.5;
    if (has_biases_) fill_value += 0.01;
    std::fill(output.data.begin(), output.data.end(), fill_value);
  }

  Shape get_output_shape() override {
    if (!configured_) {
      throw std::runtime_error("ConvolutionLayerMock: Not configured, cannot get output shape.");
    }
    return output_shape_computed_;
  }

  std::string get_type_name() const override {
    return "ConvolutionLayerMock";
  }
};

#endif 
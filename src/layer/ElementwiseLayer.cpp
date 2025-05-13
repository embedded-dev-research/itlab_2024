#ifndef ACL_ELEMENTWISE_LAYER_MOCK_H
#define ACL_ELEMENTWISE_LAYER_MOCK_H

#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "./layer/layer.h"
#include "./tensor/tensor.h"

enum class ElementwiseOp {
  ADD,
  SUB,
  MUL,
  DIV,
  MAX,
  MIN,
  SQUARED_DIFF
};

class ElementwiseLayerMock : public Layer {
 private:
  ElementwiseOp op_type_;
  Shape common_shape_;
  bool configured_ = false;

 public:
  ElementwiseLayerMock(int id, ElementwiseOp op) : op_type_(op) {
     setID(id);
  }
  void configure(const Shape& input1_shape, const Shape& input2_shape, Shape& output_shape_ref) {
    if (input1_shape.dimensions != input2_shape.dimensions || input1_shape.total_elements != input2_shape.total_elements) {
      throw std::runtime_error("ElementwiseMock: Input shapes must match for this mock.");
    }
    common_shape_ = input1_shape;
    output_shape_ref = common_shape_;
    configured_ = true;
  }

  void exec(const Tensor<double>& input, Tensor<double>& output) override {
    if (!configured_) {
      throw std::runtime_error("ElementwiseLayerMock: Layer not configured before exec.");
    }
    if (input.shape.dimensions != common_shape_.dimensions || input.shape.total_elements != common_shape_.total_elements) {
      throw std::runtime_error("ElementwiseLayerMock: Input shape mismatch in exec.");
    }
    if (output.shape.dimensions != common_shape_.dimensions || output.shape.total_elements != common_shape_.total_elements) {
      throw std::runtime_error("ElementwiseLayerMock: Output shape mismatch in exec.");
    }

    double fill_value_offset = 0.0;
    switch (op_type_) {
      case ElementwiseOp::ADD:
        fill_value_offset = 10.0;
        break;
      case ElementwiseOp::MUL:
        fill_value_offset = 20.0;
        break;
      case ElementwiseOp::MAX:
        fill_value_offset = 30.0;
        break;
      case ElementwiseOp::MIN:
        fill_value_offset = -10.0;
        break;
      case ElementwiseOp::SQUARED_DIFF:
        fill_value_offset = 5.0;
        break;
      default:
        fill_value_offset = 1.0;
        break;
    }
    std::fill(output.data.begin(), output.data.end(), static_cast<double>(getID()) + fill_value_offset + 0.3);
  }

  Shape get_output_shape() override {
    if (!configured_) {
      throw std::runtime_error("ElementwiseLayerMock: Layer not configured to get output shape.");
    }
    return common_shape_;
  }

  std::string get_type_name() const override {
    std::string op_name;
    switch (op_type_) {
      case ElementwiseOp::ADD:
        op_name = "Add";
        break;
      case ElementwiseOp::SUB:
        op_name = "Sub";
        break;
      case ElementwiseOp::MUL:
        op_name = "Mul";
        break;
      case ElementwiseOp::DIV:
        op_name = "Div";
        break;
      case ElementwiseOp::MAX:
        op_name = "Max";
        break;
      case ElementwiseOp::MIN:
        op_name = "Min";
        break;
      case ElementwiseOp::SQUARED_DIFF:
        op_name = "SquaredDiff";
        break;
      default:
        op_name = "UnknownOp";
        break;
    }
    return "Elementwise" + op_name + "LayerMock";
  }
};

#endif
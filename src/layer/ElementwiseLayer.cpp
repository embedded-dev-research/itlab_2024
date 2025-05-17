#ifndef ACL_ELEMENTWISE_LAYER_H
#define ACL_ELEMENTWISE_LAYER_H

#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "include/layer/layer.h"

using namespace arm_compute;
using namespace utils;

enum class ElementwiseOp {
    ADD,
    DIV,
    ABS,
    SIGM,
    SWISH,
    SQUARED_DIFF
};

class ElementwiseLayer : public Layer {
private:
    ElementwiseOp op_type_;
    TensorShape input1_shape, input2_shape;
    TensorShape output_shape;
    bool configured_ = false;

public:
    ElementwiseLayer(int id, ElementwiseOp op) : op_type_(op) { setID(id); }

    ElementwiseLayer() : ElementwiseLayer(0, ElementwiseOp::ADD) {}

    void configure(const TensorShape& input_shape, TensorShape& output_shape_) {
        input1_shape = input_shape;
        output_shape = input_shape;
        configured_ = true;
    }

    void configure(const TensorShape& input1_shape_, const TensorShape& input2_shape_, TensorShape& output_shape_) {
        if (input1_shape.total_size() != input2_shape.total_size()) {
            throw std::runtime_error(
                "ElementwiseLayer: Input shapes must have same total size");
        }

        input1_shape = input1_shape_;
        input2_shape = input2_shape_;
        output_shape = output_shape_;
        configured_ = true;
    }

   void exec(Tensor& input, Tensor& output) override {
      if (!configured_) {
        throw std::runtime_error(
            "ElementwiseLayer: Layer not configured before exec.");
      }

      input.allocator()->init(TensorInfo(input1_shape, 1, DataType::F32));
      output.allocator()->init(TensorInfo(output_shape, 1, DataType::F32));

      input.allocator()->allocate();
      output.allocator()->allocate();

      switch (op_type_) {
        case ElementwiseOp::ABS: {
          NEActivationLayer abs;
          abs.configure(&input, &output, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::ABS));
          abs.run();
          break;
        }
        case ElementwiseOp::SIGM: {
          NEActivationLayer sigm;
          sigm.configure(&input, &output, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC));
          sigm.run();
          break;
        }
        case ElementwiseOp::SWISH: {
          NEActivationLayer swish;
          swish.configure(&input, &output, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::SWISH));
          swish.run();
          break;
        }
        default:
          throw std::runtime_error(
              "ElementwiseLayer: This operation requires two inputs");
      }
    }

    void exec(Tensor& input1, Tensor& input2, Tensor& output) {
        if (!configured_) {
            throw std::runtime_error(
                "ElementwiseLayer: Layer not configured before exec.");
        }

        input1.allocator()->init(TensorInfo(input1_shape, 1, DataType::F32));
        input2.allocator()->init(TensorInfo(input2_shape, 1, DataType::F32));
        output.allocator()->init(TensorInfo(output_shape, 1, DataType::F32));

        input1.allocator()->allocate();
        input2.allocator()->allocate();
        output.allocator()->allocate();

        switch (op_type_) {
            case ElementwiseOp::ADD: {
              NEArithmeticAddition add;
              add.configure(&input1, &input2, &output, ConvertPolicy::WRAP);
              add.run();
              break;
            }
            case ElementwiseOp::DIV: {
              NEElementwiseDivision div;
              div.configure(&input1, &input2, &output);
              div.run();
              break;
            }
            case ElementwiseOp::SQUARED_DIFF: {
              NEElementwiseSquaredDiff sqdiff;
              sqdiff.configure(&input1, &input2, &output);
              sqdiff.run();
              break;
            }
            default:
              throw std::runtime_error(
                  "ElementwiseLayer: This operation requires single input");
        }
    }

    std::string get_type_name() const override {
        switch (op_type_) {
            case ElementwiseOp::ADD: return "ElementwiseAddLayer";
            case ElementwiseOp::DIV: return "ElementwiseDivLayer";
            case ElementwiseOp::ABS: return "ElementwiseAbsLayer";
            case ElementwiseOp::SIGM: return "ElementwiseSigmoidLayer";
            case ElementwiseOp::SWISH: return "ElementwiseSwishLayer";
            case ElementwiseOp::SQUARED_DIFF: return "ElementwiseSquaredDiffLayer";
            default:return "ElementwiseUnknownLayer";
        }
    }
};

#endif
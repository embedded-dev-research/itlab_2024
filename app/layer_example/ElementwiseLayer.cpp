#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "utils/Utils.h"

#include <iostream>
using namespace arm_compute;
using namespace utils;

class ElementwiseLayer {
    const int input_width = 5;
    const int input_height = 5;

    Tensor input1, input2, output;

public:
    void fill() {
        input1.allocator()->init(TensorInfo(TensorShape(input_width, input_height, 1), 1, DataType::F32));
        input2.allocator()->init(TensorInfo(TensorShape(input_width, input_height, 1), 1, DataType::F32));
        output.allocator()->init(TensorInfo(TensorShape(input_width, input_height, 1), 1, DataType::F32));

        input1.allocator()->allocate();
        input2.allocator()->allocate();
        output.allocator()->allocate();

        fill_random_tensor(input1, 0.f, 1.f);
        fill_random_tensor(input2, 0.f, 1.f);
    }

    void SquaredDiff() {
        NEElementwiseSquaredDiff elementwise;
        elementwise.configure(&input1, &input2, &output);
        elementwise.run();
    }

    void Division() {
        NEElementwiseDivision elementwise;
        elementwise.configure(&input1, &input2, &output);
        elementwise.run();
    }

    void Addition() {
        NEArithmeticAddition add;
        add.configure(&input1, &input2, &output, ConvertPolicy::WRAP);
        add.run();
    }

    void Swish() {
        NEActivationLayer act;
        act.configure(&input1, &input2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::SWISH));
        act.run();
    }

    void Abs() {
        NEActivationLayer act;
        act.configure(&input1, &input2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::ABS));
        act.run();
    }

    void Sigmoid() {
        NEActivationLayer act;
        act.configure(&input1, &input2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LOGISTIC));
        act.run();
    }

    void print() {
        output.print(std::cout);
    }
};

int main() {
    ElementwiseLayer a;
    a.fill();
    a.Addition();
    a.print();

    return 0;
}
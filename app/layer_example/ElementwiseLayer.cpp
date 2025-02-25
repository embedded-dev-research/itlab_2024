#include "arm_compute/runtime/NEON/functions/NEElementwiseOperations.h"
#include "utils/Utils.h"

#include <iostream>
using namespace arm_compute;
using namespace utils;

int main() {
    const int input_width = 5;
    const int input_height = 5;

    Tensor input1, input2, output;

    input1.allocator()->init(TensorInfo(TensorShape(input_width, input_height, 1), 1, DataType::F32));
    input2.allocator()->init(TensorInfo(TensorShape(input_width, input_height, 1), 1, DataType::F32));
    output.allocator()->init(TensorInfo(TensorShape(input_width, input_height, 1), 1, DataType::F32));

    input1.allocator()->allocate();
    input2.allocator()->allocate();
    output.allocator()->allocate();

    fill_random_tensor(input1, 0.f, 1.f);
    fill_random_tensor(input2, 0.f, 1.f);

    NEElementwiseSquaredDiff elementwise;
    elementwise.configure(&input1, &input2, &output);

    elementwise.run();

    output.print(std::cout);
}

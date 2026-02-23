#include <iostream>
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "utils/Utils.h"

using namespace arm_compute;
using namespace utils;

int main() {
    Tensor input;
    const int input_width = 3;
    const int input_height = 3;
    const int channels = 2;
    const int axis = 2;

    input.allocator()->init(TensorInfo(TensorShape(input_width, input_height, channels), 1, DataType::F32));
    input.allocator()->allocate();
    fill_random_tensor(input, 0.f, 1.f);

    Tensor output1, output2;
    std::vector<ITensor*> outputs = { &output1, &output2 };

    NESplit split;
    split.configure(&input, outputs, axis);

    output1.allocator()->allocate();
    output2.allocator()->allocate();

    split.run();

    output1.print(std::cout);
    output2.print(std::cout);

    return 0;
}
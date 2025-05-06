#include <iostream>
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "utils/Utils.h"

using namespace arm_compute;
using namespace utils;

int main() {
    Tensor input1, input2;
    Tensor output;
    std::vector<const ITensor *> input;

    const int input_width  = 3;
    const int input_height = 3;
    const int axis = 2;
    
    input1.allocator()->init(TensorInfo(TensorShape(input_width, input_height, 1), 1, DataType::F32));
    input2.allocator()->init(TensorInfo(TensorShape(input_width, input_height, 1), 1, DataType::F32));

    input1.allocator()->allocate();
    input2.allocator()->allocate();

    fill_random_tensor(input1, 0.f, 1.f);
    fill_random_tensor(input2, 0.f, 1.f);
    
    input.push_back(&input1);
    input.push_back(&input2);

    NEConcatenateLayer concat; 
    concat.configure(input, &output, axis);
    output.allocator()->allocate();

    concat.run();

    output.print(std::cout);
}
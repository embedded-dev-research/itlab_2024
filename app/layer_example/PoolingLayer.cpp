#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "utils/Utils.h"

#include <iostream>
using namespace arm_compute;
using namespace utils;

int main() {
    Tensor input;
    Tensor output;
    
    const int input_width  = 5;
    const int input_height = 5;

    input.allocator()->init(TensorInfo(TensorShape(input_width, input_height, 1), 1, DataType::F32));
    output.allocator()->init(TensorInfo(TensorShape(input_width, input_height, 1), 1, DataType::F32));
    
    input.allocator()->allocate();
    output.allocator()->allocate();
    
    fill_random_tensor(input, 0.f, 1.f);

    NEPoolingLayer pool; 
    pool.configure(&input, &output, PoolingLayerInfo(PoolingType::MAX,  DataLayout::NHWC));
    pool.run();

    output.print(std::cout);
}
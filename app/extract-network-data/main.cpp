
#include <iostream>
#include <vector>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>

int main()
{
    std::string modelPath = cv::samples::findFile("../yolov8s.onnx");
    cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);
    std::vector<std::string> layerNames = net.getLayerNames();

    for (auto& name : layerNames)
    {
        std::cout << name << std::endl;
    }

    return 0;
}

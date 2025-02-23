
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[fatal error] Path to an input image is missing\n";
    return 1;
  }

  // read an input image
  cv::Mat raw_image = cv::imread(argv[1]);
  if (raw_image.empty()) {
    std::cerr << "[fatal error] Failed to read the input image\n";
    return 1;
  }

  // read the classification list
  std::ifstream file("classification_list.txt");

  if (!file.is_open()) {
    std::cerr << "[fatal error] Failed to open the classification list file\n";
    return 1;
  }

  std::string line;
  std::vector<std::string> class_list;
  while (std::getline(file, line)) {
    class_list.push_back(line);
  }

  // read the network model
  cv::dnn::Net net = cv::dnn::readNetFromONNX("yolov8n-cls.onnx");
  if (net.empty()) {
    std::cerr << "[fatal error] Failed to read the network model\n";
    return 1;
  }

  // prepare the image for inference
  cv::Mat input_image;
  cv::resize(raw_image, input_image, cv::Size(224, 224), cv::INTER_LINEAR);
  cv::Mat blob = cv::dnn::blobFromImage(input_image, 1.0 / 255.0,
                                        cv::Size(224, 224), cv::Scalar());

  // inference
  net.setInput(blob);
  cv::Mat output = net.forward("output0");

  // extract a result from the output
  double max_class_score = 0.0;
  cv::Point max_loc;
  cv::minMaxLoc(output, nullptr, &max_class_score, nullptr, &max_loc);

  // print the results
  std::cout << "class ID: " << max_loc.x << '\n';
  std::cout << "class name: " << class_list[max_loc.x] << '\n';
  std::cout << "confidence: " << max_class_score << '\n';

  return 0;
}

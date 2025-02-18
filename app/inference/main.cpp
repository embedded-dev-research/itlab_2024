
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


size_t readClassList(std::vector<std::string>& class_list, const std::string& file_path) {
  std::ifstream file(file_path);

  if (!file.is_open()) {
    return 0;
  }

  std::string line;
  while (std::getline(file, line)) {
    class_list.push_back(line);
  }

  return class_list.size();
}

int main() { 

  std::vector<std::string> class_list;
  size_t result = readClassList(class_list, "resource/classification_list.txt");
  if (result == 0) {
    std::cout << "Fatal error: failed to read classification list" << std::endl;
    return 1;
  }

  std::string model_path = cv::samples::findFile("resource/resnet50.onnx");
  cv::dnn::Net net = cv::dnn::readNet(model_path);

  cv::Mat img = cv::imread("resource/car.jpg");
  if (img.empty()) {
    std::cout << "Fatal error: failed to read image" << std::endl;
  }

  cv::Mat blob = cv::dnn::blobFromImage(img, 0.003921569, cv::Size(224, 224), cv::Scalar(104.0, 117.0, 123.0));
  net.setInput(blob);

  cv::Mat prob = net.forward();

  int class_id;
  double confidence;
  cv::Point class_id_point;
  cv::minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &class_id_point);
  class_id = class_id_point.x;

  std::cout << "Class #" << class_id << ": " << class_list[class_id] << " {" << confidence << "}" << std::endl;

  return 0;
}


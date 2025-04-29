
#include "model_parser/onnx_model_parser.h"

#include <onnx.pb.h>

#include <stack>

#include "layer/layer.h"

ONNX_ModelParser::ONNX_ModelParser(const std::string& filename) {
  m_filename = filename;
}

Graph ONNX_ModelParser::Parse() {
  Graph result_graph;

  if (!LoadModel(m_filename)) {
    std::cerr << "Failed to open a model file" << std::endl;
    return result_graph;
  }

  const auto& graph = m_model.graph();

  for (const auto& tensor : graph.initializer()) {
    m_weights[tensor.name()] = &tensor;
  }

  // image: node name -> layer
  std::unordered_map<std::string, std::shared_ptr<Layer>> node_to_layer;

  // create layers for every node
  for (const auto& node : graph.node()) {
    std::string layer_name = node.name();
    auto any_layer = std::make_shared<AnyLayer>(layer_name);

    // parse weights and bias
    ParseWeightsAndBias(node, any_layer);

    // parse attributes
    ParseAttributes(node, any_layer->Attributes);

    auto layer = std::dynamic_pointer_cast<Layer>(any_layer);
    result_graph.addLayer(layer);
    node_to_layer[layer_name] = layer;
  }

  // bypass nodes and create edges
  for (const auto& node : graph.node()) {
    std::string layer_name = node.name();
    auto current_layer = node_to_layer[layer_name];

    for (const auto& input_name : node.input()) {
      auto it = node_to_layer.find(input_name);
      if (it != node_to_layer.end()) {
        result_graph.addEdge(it->second, current_layer);
      }
    }
  }

  return result_graph;
}

bool ONNX_ModelParser::LoadModel(const std::string& filename) {
  // open a model file
  std::ifstream model_file(filename, std::ios::binary);
  if (!model_file.is_open()) {
    return false;
  }

  // parse the file to the onnx model
  if (!m_model.ParseFromIstream(&model_file)) {
    return false;
  }

  model_file.close();

  return true;
}

bool ONNX_ModelParser::ParseAttributes(
    const onnx::NodeProto& node,
    std::unordered_map<std::string, std::string>& attributes) {
  for (int i = 0; i < node.attribute_size(); ++i) {
    const auto& attrib = node.attribute(i);

    std::string attrib_name = attrib.name();
    std::string attrib_data;

    switch (attrib.type()) {
      case onnx::AttributeProto_AttributeType_STRING:
        attrib_data = attrib.s();
        break;

      case onnx::AttributeProto_AttributeType_FLOAT:
        attrib_data = std::to_string(attrib.f());
        break;

      case onnx::AttributeProto_AttributeType_INT:
        attrib_data = std::to_string(attrib.i());
        break;

      case onnx::AttributeProto_AttributeType_STRINGS:
        attrib_data += '[';

        int si;
        for (si = 0; si < attrib.floats_size() - 1; ++si) {
          attrib_data += attrib.strings(si) + ", ";
        }

        attrib_data += attrib.strings(si) + ']';
        break;

      case onnx::AttributeProto_AttributeType_FLOATS:
        attrib_data += '[';

        int fi;
        for (fi = 0; fi < attrib.floats_size() - 1; ++fi) {
          attrib_data += std::to_string(attrib.floats(fi)) + ", ";
        }

        attrib_data += std::to_string(attrib.floats(fi)) + ']';
        break;

      case onnx::AttributeProto_AttributeType_INTS:
        attrib_data += '[';

        int ii;
        for (ii = 0; ii < attrib.ints_size() - 1; ++ii) {
          attrib_data += std::to_string(attrib.ints(ii)) + ", ";
        }

        attrib_data += std::to_string(attrib.ints(ii)) + ']';
        break;
    }

    attributes.insert({attrib_name, attrib_data});
  }

  return true;
}

bool ONNX_ModelParser::ParseWeightsAndBias(
    const onnx::NodeProto& node, const std::shared_ptr<AnyLayer>& any_layer) {
  for (const auto& input_name : node.input()) {
    auto weight_it = m_weights.find(input_name);
    if (weight_it != m_weights.end()) {
      const auto* tensor_ptr = weight_it->second;

      const char* raw_data = tensor_ptr->raw_data().data();
      size_t data_size = tensor_ptr->raw_data().size() / sizeof(float);

      if (input_name.find("weight")) {
        any_layer->Weights = std::vector<float>(raw_data, raw_data + data_size);
      } else if (input_name.find("bias")) {
        any_layer->Bias = std::vector<float>(raw_data, raw_data + data_size);
      }
    }
  }

  return true;
}

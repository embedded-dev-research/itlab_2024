#include "IR.h"

IR::IR(std::string filename, Model model): _graph(){
    _fileName = filename;
    _model = model;
    switch (_model)
    {
    case Model::ONNX:
        _modelParser = new ONNX_ModelParser(_fileName);
    break;

    case Model::OPENCV:
        _modelParser = new OPENCV_ModelParser(_fileName);
    break;

    // case Model::PYTORCH:
    //     _modelParser = new OPENCV_ModelParser(_fileName);
    //     break;
    default:
        break;
    }

    _graph = _modelParser->Parse();
}

const bool IR::operator==(const Graph& const) const {
    return 0;
}


IR::~IR(){
    delete _modelParser;
    _modelParser = nullptr;
}
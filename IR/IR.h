#include "graph.h"
#include "map"

class AnyLayer : protected Layer
{
private:
    std::string _name;
    std::map<std::string, std::string> _attribute;
    std::vector<float> weights;
    std::vector<float> bias;   

public:
    AnyLayer();
    ~AnyLayer() = default; 

    void addNeighbor(Layer* neighbor);
    void removeNeighbor(Layer* neighbor);
    std::list<Layer*> neighbors_; 
};



enum Model{ONNX, YOLO, OPENCV};

class IR
{
private:
    std::string _fileName;
    Graph _graph;
    Model _model;

    void _ONNX_modelParser(std::string filename);
    void _YOLO_modelParser(std::string filename);
    void _OPENCV_modelParser(std::string filename);
public:

    IR(std::string filename, Model model): _graph(){
        _fileName = filename;
        _model = model;
        switch (_model)
        {
        case Model::ONNX:
            _ONNX_modelParser(_fileName);
            break;

        case Model::OPENCV:
            _OPENCV_modelParser(_fileName);
            break;

        case Model::YOLO:
            _YOLO_modelParser(_fileName);
            break;

        default:
            break;
        }
    }

    void ChangeModelType(Model model){
        _model = model;
    }
        
    void ChangeFile(std::string filename){
        _fileName = filename;
    }

    ~IR() = default;
};


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

enum Model{ONNX, PYTORCH, OPENCV};

class ModelParser {
protected:
    std::string _fileName;

public:
    explicit ModelParser(std::string filename) : _fileName(std::move(filename)) {}
    virtual ~ModelParser() = default;
    virtual Graph Parse() = 0;
};

class ONNX_ModelParser : public ModelParser {
public:
    explicit ONNX_ModelParser(std::string filename) : ModelParser(std::move(filename)) {}
    Graph Parse() override;
};

class OPENCV_ModelParser : public ModelParser {
public:
    explicit OPENCV_ModelParser(std::string filename) : ModelParser(std::move(filename)) {}
    Graph Parse() override; 
};

// class PYTORCH_ModelParser : ModelParser{
// public:
    // explicit OPENCV_ModelParser(std::string filename) : ModelParser(std::move(filename)) {}
    // Graph Parse() override; 
// };


class IR
{
private:
    std::string _fileName;
    Graph _graph;
    Model _model;
    ModelParser* _modelParser;

public:

    IR(std::string filename, Model model);

    const bool operator==(const Graph& graph) const;

    // void ChangeModelType(Model model){
    //     _model = model;
    // }
        
    // void ChangeFile(std::string filename){
    //     _fileName = filename;
    // }

    ~IR();
};


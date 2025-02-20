#!/bin/bash

python3 -m venv env
source env/bin/activate
pip install ultralytics 
yolo export model=yolo11x.pt format=onnx  
deactivate
rm -rf env
rm -rf yolo11x.pt

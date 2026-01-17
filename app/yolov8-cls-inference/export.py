
from ultralytics import YOLO

# load the YOLOv8-cls model
model = YOLO("yolov8n-cls.pt")

# export the classification list
with open('classification_list.txt', 'w') as file:
    for name in model.names.values():
        file.write(name + '\n')

# export the model to ONNX format
model.export(format="onnx", imgsz=224)

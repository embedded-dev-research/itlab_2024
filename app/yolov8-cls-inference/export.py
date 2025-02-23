
from ultralytics import YOLO

# load the YOLOv8-cls model
model = YOLO("yolov8n-cls.pt")

# export the model to ONNX format
model.export(format="onnx", imgsz=224)
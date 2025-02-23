
## YOLOv8-cls inference

1. OpenCV installation

   [Get started with OpenCV](https://opencv.org/get-started/)

2. Ultralytics installation

   Create and activate python venv
   ```
   python -m venv <venv dir>
   source <venv dir>/bin/activate
   ```
   Install ultralytics using pip
   ```
   pip install ultralytics
   ```

3. Export YOLOv8-cls model
   ```
   python export.py
   ```

4. Run the project 
   ```
   make -C build
   build/yolov8-cls-inference <image path>
   ```

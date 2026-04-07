import os
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train('dataset.yaml', epochs = 100, imgsz = 640)





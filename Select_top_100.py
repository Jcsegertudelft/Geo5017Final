import os
from ultralytics import YOLO
import numpy as np

model = YOLO("runs/detect/train4/weights/best.pt")

i = 0
max_conf = []

for img in os.listdir("images/test"):
    result = model([os.path.join("images/test",img)])[0]
    conf = result.boxes.conf
    if len(conf) != 0:
        max_conf.append(float(np.max(conf)))
    else:
        max_conf.append(0.)

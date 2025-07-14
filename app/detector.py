from ultralytics import YOLO
import numpy as np
import torch

class YoloDetector:
    def __init__(self, model_path, conf_thres=0.25):
        self.model = YOLO(model_path)
        self.conf_thres = conf_thres

    def predict(self, frame):
        # Chạy YOLOv8 inference
        results = self.model.predict(
            source=frame,
            conf=self.conf_thres,
            verbose=False
        )
        boxes = results[0].boxes

        bboxes = []
        confidences = []

        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()  # (x1, y1, x2, y2)
            x1, y1, x2, y2 = xyxy
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2]) 

            w = x2 - x1
            h = y2 - y1

            bboxes.append([x1, y1, w, h])               # DeepSORT dùng [x, y, w, h]
            confidences.append(float(box.conf[0]))     # Convert về float Python

        return bboxes, confidences


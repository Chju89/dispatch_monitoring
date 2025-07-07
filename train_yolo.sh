#!/bin/bash

# Train YOLOv8n on Quadro P1000 with safe settings
yolo task=detect mode=train \
  model=yolov8n.pt \
  data=/home/trieu-nguyen/MLOps/dispatch_monitoring/data/raw/Dataset/Detection/dataset.yaml \
  epochs=100 \
  patience=20 \
  imgsz=320 \
  batch=1 \
  device=0 \
  lr0=0.002 \
  warmup_epochs=5 \
  optimizer=SGD \
  name=yolov8n_dispatch_train_safe


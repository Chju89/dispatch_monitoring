# export_models.py

import torch
from ultralytics import YOLO

# === Export YOLOv8 ===
print("🚀 Exporting YOLOv8 model to ONNX...")
yolo_model = YOLO("models/detection/best.pt")
yolo_model.export(format="onnx", imgsz=1088)
print("✅ YOLOv8 exported: best.onnx")

# === Export ResNet18 ===
print("🚀 Exporting ResNet18 model to ONNX...")
resnet_model = torch.load("models/classification/resnet18_dispatch.pt", map_location="cpu")
resnet_model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
output_path = "models/classification/resnet18_dispatch.onnx"
torch.onnx.export(
    resnet_model,
    dummy_input,
    f=str(output_path),
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)
print("✅ ResNet18 exported: resnet18_dispatch.onnx")


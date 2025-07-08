import mlflow
import os

# 🔗 MLflow tracking server via ngrok (cập nhật URL mới nhất bạn đang dùng)
mlflow.set_tracking_uri("https://c7748eb5e819.ngrok-free.app")
mlflow.set_experiment("dispatch-pipeline")

# 🧠 Log YOLOv8 detection model
yolo_path = "runs/detect/yolov8n_dispatch_colab"
if os.path.exists(yolo_path):
    with mlflow.start_run(run_name="relog_yolov8n"):
        mlflow.log_param("source", "Colab")
        mlflow.log_param("model", "yolov8n")
        mlflow.log_param("type", "object_detection")
        mlflow.log_artifacts(yolo_path)
else:
    print(f"❌ YOLOv8 folder not found at {yolo_path}")

# 🧠 Log ResNet18 classifier
clf_path = "models/resnet18_dispatch.pt"
if os.path.exists(clf_path):
    with mlflow.start_run(run_name="relog_resnet18"):
        mlflow.log_param("source", "Colab")
        mlflow.log_param("model", "resnet18")
        mlflow.log_param("type", "classifier")
        mlflow.log_artifact(clf_path)
else:
    print(f"❌ Classifier model not found at {clf_path}")


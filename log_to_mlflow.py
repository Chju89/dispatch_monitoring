import mlflow

mlflow.set_experiment("dispatch-pipeline")

with mlflow.start_run(run_name="relog_yolov8n"):
    mlflow.log_param("source", "Colab")
    mlflow.log_param("model", "yolov8n")
    mlflow.log_artifacts("runs/detect/yolov8n_dispatch_colab")

with mlflow.start_run(run_name="relog_resnet18"):
    mlflow.log_param("source", "Colab")
    mlflow.log_param("model", "resnet18")
    mlflow.log_artifact("models/resnet18_dispatch.pt")

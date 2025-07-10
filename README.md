# 🚚 Dispatch Monitoring System

An end-to-end MLOps pipeline to monitor the kitchen dispatch area by detecting, tracking, and classifying dishes and trays directly on video. Integrated feedback loop enables continuous improvement of the classification model.

---

## 🌟 Features

- 🧠 **Object Detection**: YOLOv8-based custom model
- 🛰️ **Object Tracking**: DeepSORT to maintain consistent IDs across frames
- 🧪 **Classification**: ResNet18 for classifying dishes/trays into:
  - `dish/empty`, `dish/not_empty`, `dish/kakigori`
  - `tray/empty`, `tray/not_empty`, `tray/kakigori`
- 📋 **Frame-level Logging**: Results saved in structured `.pkl`/`.csv` for later use
- 🖼️ **Feedback UI**: Interactive Streamlit interface for reviewing and correcting model predictions
- 🔁 **Retraining Pipeline**: Leverages user feedback to fine-tune classifier
- 📊 **MLflow Integration**: Logs training runs, artifacts, and metrics for all models
- 🐳 **Dockerized**: Easily deploy app and MLflow with Docker Compose

---

## 🔄 Pipeline Overview (Updated)

```mermaid
graph TD
    A[Input Video] --> B[Tracking + Detection (YOLOv8 + DeepSORT)]
    B --> C[Frame-wise Classification (ResNet18)]
    C --> D[Log Results (CSV + PKL)]
    D --> E[Feedback UI (Streamlit)]
    E --> F[Feedback Data]
    F --> G[Retraining Script]
```

---

## 📁 Directory Overview

```
.
├── app.py                         # Streamlit app for feedback
├── data/
│   ├── raw/                       # Input video, frames, dataset
│   ├── processed/tracking/       # Tracking & classification logs
│   ├── feedback/                 # JSON + Pickle feedback data
│   └── retrain/                  # Data for retraining classifiers/detectors
├── models/                       # Saved models
│   ├── detection/best.pt
│   └── classification/resnet18_dispatch.pt
├── notebooks/                    # Colab notebooks for training & inference
│   ├── tracking_on_colab.ipynb
│   ├── train_classifier_colab.ipynb
│   └── train_yolov8_colab.ipynb
├── scripts/                      # Pipeline helper scripts
├── src/
│   ├── tracking.py               # Runs YOLO + DeepSORT tracking
│   ├── classify.py               # Applies classification to tracked objects
│   ├── retrain.py                # Fine-tunes model from feedback
│   └── feedback_ui/             # Streamlit UI logic
├── docker-compose.yaml          # Orchestrates app + mlflow services
├── Dockerfile.app               # Build file for Streamlit app
├── Dockerfile.mlflow            # Build file for MLflow server
├── mlruns/, mlartifacts/        # MLflow tracking and artifacts
├── requirements.txt, environment.yaml
└── README.md
```

---

## 🛠️ How to Run (Local)

### 1. Run tracking + classification on a video
```bash
python src/tracking.py        # Output: bbox_tracking_log.pkl
python src/classify.py        # Output: classification_result.csv, ui_objects.pkl
```

### 2. Launch feedback UI
```bash
streamlit run app.py
```

### 3. Retrain classifier with feedback
```bash
python src/retrain.py
```

---

## 🐳 Run with Docker

### 1. Build and run services
```bash
docker-compose up --build
```

### 2. Access services
- Streamlit UI: [http://localhost:8501](http://localhost:8501)
- MLflow UI: [http://localhost:5000](http://localhost:5000)

> Make sure `models/`, `data/`, `src/`, `scripts/` are all in project root.


### 3. Stop Docker
```bash
docker-compose down --volumes --remove-orphans
docker image prune -f
```
---

## 📤 Outputs

| Step            | Output File/Path                                |
|----------------|--------------------------------------------------|
| Tracking Log    | `data/processed/tracking/bbox_tracking_log.pkl` |
| Classification  | `data/processed/tracking/classification_result.csv` |
| UI Objects Log  | `data/processed/tracking/ui_objects.pkl`         |
| User Feedback   | `data/feedback/feedback.pkl` or `.json`          |
| Retrain Artifacts | `data/retrain/classifier/`                     |

---

## 📊 MLflow Runs

| Model                     | Run Name / Folder                  |
|---------------------------|------------------------------------|
| YOLOv8 Detection          | `runs/detect/yolov8n_dispatch_colab`|
| ResNet18 Classification  | `runs/classify/resnet18_dispatch.pt`|
| Feedback Retrain          | Tracked in `mlruns/`, `mlartifacts/`

---

## 👤 Author

**Nguyen Quang Trieu**  
MLOps Enthusiast | AI Learner | Senior Rigging Artist | Former Physics Teacher

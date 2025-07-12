# 🚚 Dispatch Monitoring System (Realtime Version)

A production-ready, **real-time monitoring system** for kitchen dispatch areas using object detection, tracking, and classification directly on video. Includes feedback loop, model retraining, and full MLOps integration with **MLflow**.

---

## 🌟 Features

- 🧠 **Object Detection**: YOLOv8 custom model (e.g., `yolov8m_dispatch_colab`)
- 🛰️ **Object Tracking**: DeepSORT with consistent tracking IDs
- 🧪 **Classification**: ResNet18 classifier with 6 dish/tray states:
  - `dish_empty`, `dish_not_empty`, `dish_kakigori`  
  - `tray_empty`, `tray_not_empty`, `tray_kakigori`
- 🎯 **Realtime Inference**: End-to-end processing per video frame
- 🧾 **Interactive Feedback**:
  - View frame-by-frame detection results
  - Delete any incorrect object by ID
  - Adjust classification via dropdowns
- 🔁 **Feedback Loop**: Persist changes to JSON and enable retraining
- 📊 **MLflow Integration**: Tracks model training, evaluation, metrics & artifacts
- 🐳 **Dockerized Deployment**: Fully containerized with Docker Compose

---

## 🔄 Pipeline Overview

```mermaid
graph TD
    A[Input Video (mp4)] --> B[Detection + Tracking (YOLOv8 + DeepSORT)]
    B --> C[Realtime Classification (ResNet18)]
    C --> D[Draw & Show Bounding Boxes]
    D --> E[Feedback UI (Streamlit)]
    E --> F[Export to Feedback Logs (JSON)]
    F --> G[Retraining (optional)]
```

---

## 📁 Directory Structure

```
.
├── app.py                         # Streamlit app (now real-time)
├── models/                        # Trained models
│   ├── detection/best.pt          # YOLOv8 model
│   └── classification/resnet18_dispatch.pt  # Full ResNet model
├── data/
│   ├── raw/video_shortened.mp4    # Input video
│   └── feedback/                  # User feedback in JSON
├── src/
│   ├── tracking.py                # (legacy) tracking script
│   ├── classify.py                # (legacy) classification script
│   ├── retrain.py                 # Classifier retraining using feedback
├── notebooks/
│   ├── train_classifier_colab.ipynb
│   └── train_yolov8_colab.ipynb
├── requirements.txt, environment.yaml
├── docker-compose.yaml
├── Dockerfile.app, Dockerfile.mlflow
└── README.md
```

---

## 🚀 Run Locally

### 1. Run the App
```bash
streamlit run app.py
```

### 2. Provide Video Input
- Drop your video in `data/raw/video_shortened.mp4` (or change path in code)

### 3. Classify Frame-by-Frame
- Realtime detection + classification
- Delete wrong objects, correct status via dropdown
- Press **"Apply Change"** to log your feedback

---

## 🧪 Retraining

After collecting enough feedback:

```bash
python src/retrain.py
```

> Your corrected feedback is saved in `data/feedback/*.json`, and can be used to update the classifier.

---

## 🧠 MLflow Integration

- Automatically logs model params, metrics, and artifacts
- URLs:
  - Streamlit UI: http://localhost:8501
  - MLflow UI: http://localhost:5000

---

## 🐳 Run with Docker
To Clone
```bash
mkdir test_repo && cd test_repo
git clone https://github.com/Chju89/dispatch_monitoring.git
cd dispatch_monitoring
```
To run:
```bash
docker-compose up --build
```

To stop:
```bash
docker-compose down --volumes --remove-orphans
```

---

## 📤 Output Artifacts

| Step                | Output File                          |
|---------------------|--------------------------------------|
| Realtime Feedback   | `data/feedback/*.json`               |
| Trained Classifier  | `models/classification/resnet18_dispatch.pt` |
| Trained Detector    | `models/detection/best.pt`           |
| MLflow Artifacts    | `mlruns/`, `mlartifacts/`            |

---

## 👤 Author

**Nguyen Quang Trieu**  
MLOps Enthusiast | AI Learner | Senior Rigging Artist  
📫 [quangtrieu.sp@gmail.com](mailto:quangtrieu.sp@gmail.com)

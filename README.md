# 📦 Dispatch Monitoring System – MLOps Project

An intelligent system for detecting, tracking, and classifying items in a commercial kitchen dispatch area. Built with YOLOv8, DeepSORT, ResNet/MobileNet, and feedback retraining loop. MLflow is integrated for tracking experiments.

---

## 🎯 Features

- ✅ **Object Detection** using YOLOv8
- ✅ **Object Tracking** with DeepSORT
- ✅ **Item Classification**: `dish/tray` + `empty/not_empty/kakigori`
- ✅ **User Feedback** interface (Streamlit)
- ✅ **Retraining Pipeline** based on feedback
- ✅ **MLflow Tracking** for detection & classification
- ✅ **Dockerized**: run full pipeline with MLflow locally

---

## 📁 Project Structure

```

dispatch_monitoring/
├── data/                       # Raw + processed + feedback
│   ├── raw/
│   ├── processed/
│   └── feedback/
├── models/                    # best.pt, resnet18.pt, etc.
├── mlruns/                    # MLflow logs (optional)
├── notebooks/                 # .ipynb training on Colab
├── src/                       # All pipeline code
│   ├── detection.py
│   ├── tracking.py
│   ├── classify.py
│   ├── feedback\_ui.py
│   └── retrain.py
├── app.py                     # Run Streamlit UI for feedback
├── requirements.txt
├── log_to_mlflow.py           # Relog from Colab outputs
├── Dockerfile.app
├── Dockerfile.mlflow
├── docker-compose.yaml
└── README.md

````

---

## ⚙️ Setup Instructions

### ✅ Option 1: Docker Compose (App + MLflow)
```bash
docker compose up --build
````

* MLflow UI → [http://localhost:5000](http://localhost:5000)
* Streamlit App → [http://localhost:8501](http://localhost:8501)

---

### 🧠 Option 2: Train model on Google Colab

* Use provided notebooks:

  * `notebooks/train_yolov8_colab.ipynb`
  * `notebooks/train_classifier_colab.ipynb`

* After training, download:

  * `runs/` folder (YOLO)
  * `resnet18_dispatch.pt` (classifier)

* Use `log_to_mlflow.py` to relog artifacts:

```bash
python log_to_mlflow.py
```

---

## 🔁 Pipeline Overview

1. **Detection**: YOLOv8 detects dish/tray
2. **Tracking**: DeepSORT assigns ID
3. **Classification**: Object → ResNet18/MobileNetV2
4. **Feedback**: User reviews prediction (via UI)
5. **Retrain**: `src/retrain.py` with feedback data
6. **Tracking**: MLflow logs model versions, accuracy, feedback size

---

## 🧪 MLflow Tracking

* MLflow UI: [http://localhost:5000](http://localhost:5000)
* Logs:

  * Detection + classifier metrics
  * Params: batch, epochs, lr,...
  * Artifacts: weights, images, training curves

---

## 📝 Feedback Example (JSON)

```json
{
  "object_id": "12",
  "frame_id": "frame_1050",
  "predicted_label": "empty",
  "corrected_label": "not_empty"
}
```

---

## ✅ Improvements (Future)

* [ ] Add image augmentation to improve classifier
* [ ] Replace Streamlit with FastAPI or React UI
* [ ] Auto-retrain + redeploy using Jenkins or GitHub Actions
* [ ] Cloud deployment (GCP, AWS, GKE)

---

## 👨‍💻 Author

**Nguyễn Quang Triều**




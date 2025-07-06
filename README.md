# 📦 Dispatch Monitoring System

An intelligent monitoring system for tracking and classifying items in a commercial kitchen dispatch area. Built with YOLOv8, DeepSORT, and a classification model. Feedback loop is integrated to improve model accuracy over time using user corrections and MLflow tracking.

---

## 🚀 Features

- ✅ **Real-time object detection** using YOLOv8
- ✅ **Object tracking** with Deep SORT
- ✅ **Item classification** (`empty`, `not_empty`, `kakigori`)
- ✅ **User feedback UI** via Streamlit
- ✅ **Retrain pipeline** based on feedback
- ✅ **MLflow integration** for experiment tracking

---

## 📁 Project Structure

```

dispatch\_monitoring\_project/
│
├── data/
│   ├── raw/               # Original video/data
│   ├── processed/         # Frame-by-frame or cropped data
│   └── feedback/          # User feedback JSON/CSV
│
├── models/                # Trained detection/classification models
├── notebooks/             # Exploratory notebooks
├── src/
│   ├── detection.py       # YOLOv8 inference
│   ├── tracking.py        # DeepSORT tracking
│   ├── classify.py        # Classifier inference
│   ├── feedback\_ui.py     # Streamlit UI
│   └── retrain.py         # Retraining logic
│
├── app.py                 # Run full pipeline
├── requirements.txt
├── environment.yaml
├── Dockerfile
├── docker-compose.yml
└── README.md

````

---

## ⚙️ Installation

### 📌 Option 1: Local (Conda)

```bash
conda env create -f environment.yaml
conda activate dispatch-monitoring-env
streamlit run app.py
````

### 📌 Option 2: Docker

```bash
# Build & run using Docker
docker build -t dispatch-monitoring-app .
docker run -p 8501:8501 dispatch-monitoring-app
```

### 📌 Option 3: Docker Compose (App + MLflow)

```bash
docker-compose up --build
```

---

## 🧠 How It Works

1. **Detection**: YOLOv8 model detects trays/plates in each frame.
2. **Tracking**: DeepSORT assigns unique IDs to follow each object.
3. **Classification**: Each object is cropped and passed to a classifier.
4. **Feedback**: User corrects predictions via UI. Feedback is stored.
5. **Retrain**: When enough feedback is collected, retrain classifier using `src/retrain.py`.
6. **Track Models**: MLflow logs each version of model and metrics.

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

## 📊 MLflow Tracking

* Access MLflow UI: [http://localhost:5000](http://localhost:5000)
* Logs:

  * model versions
  * accuracy
  * number of feedback samples used for retraining

---

## 📌 Todo / Improvements

* [ ] Improve classifier performance with augmentations
* [ ] Replace Streamlit UI with web app (Flask or FastAPI)
* [ ] Add CI/CD pipeline for auto-retrain & redeploy
* [ ] Add GCP / AWS deployment option

---

## 👨‍💻 Author

**Nguyễn Quang Triều**


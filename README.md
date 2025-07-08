# Dispatch Monitoring System 🚀

A lightweight end-to-end MLOps pipeline to detect, track, and classify objects in a kitchen dispatch area, with feedback loop and retraining capabilities.

---

## 🌟 Features

* 👀 Object Detection: YOLOv8-based model to detect `dish` and `tray`
* 🛍️ Object Tracking: DeepSORT to maintain track IDs across frames
* 🧃 Classification: ResNet18 model to classify cropped objects into 6 classes:

  * `dish/empty`, `dish/not_empty`, `dish/kakigori`
  * `tray/empty`, `tray/not_empty`, `tray/kakigori`
* 📈 MLflow Integration: Logs metrics, parameters, artifacts of both models
* 📊 Feedback UI: Streamlit app allows user to correct wrong predictions
* 🔄 Retraining Script: Model improvement loop based on user feedback

---

## 🔄 Pipeline Overview

```mermaid
graph TD
    A[Input Video / Frames] --> B[Detection (YOLOv8)]
    B --> C[Tracking (DeepSORT)]
    C --> D[Crop Objects by ID]
    D --> E[Classification (ResNet18)]
    E --> F[Overlay Labels on Frame]
    F --> G[Save Video + Frames]
    G --> H[Feedback UI]
    H --> I[Feedback Log]
    I --> J[Retrain Script]
```

---

## 📁 Directory Structure

```
.
├── data/
│   ├── raw/                         # Raw video + dataset
│   ├── processed/
│   │   ├── tracking/               # Tracked frames, crops, result.csv
│   │   ├── feedback.csv            # Feedback from user
│   │   ├── output_video.mp4        # Final labeled video
├── models/
│   ├── detection/best.pt           # YOLOv8 trained model
│   ├── classification/resnet18.pt # Classifier model
├── notebooks/                      # Colab training notebooks
├── src/                            # Source code
│   ├── detection.py
│   ├── tracking.py
│   ├── classify.py
│   ├── extract_frames.py
│   ├── feedback_ui.py
│   ├── retrain.py
├── app.py                          # Streamlit app for feedback
├── requirements.txt
└── README.md
```

---

## 📚 Model Training

### 1. Train YOLOv8 (on Colab)

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='dataset.yaml', epochs=100, imgsz=640, batch=16, name='yolov8n_dispatch')
```

### 2. Train ResNet18 Classifier

```python
# Colab notebook: train_classifier_colab.ipynb
# Logs model + metrics to MLflow
```

---

## 🛠️ Usage

### 1. Extract frames

```bash
python src/extract_frames.py
```

### 2. Run detection + tracking

```bash
python src/tracking.py
```

### 3. Run classification

```bash
python src/classify.py
```

### 4. Launch Feedback UI

```bash
streamlit run app.py
```

### 5. Retrain from feedback (optional)

```bash
python src/retrain.py
```

---

## 📊 Output Files

| Step            | Output Path                                         |
| --------------- | --------------------------------------------------- |
| Frames w/ label | `data/processed/tracking/frames_with_id/`           |
| Cropped objects | `data/processed/tracking/crops/`                    |
| Classification  | `data/processed/tracking/classification_result.csv` |
| Feedback        | `data/processed/feedback.csv`                       |
| Final video     | `data/processed/output_video.mp4`                   |

---

## 🔍 MLflow Tracking

| Run Name                  | Description                  |
| ------------------------- | ---------------------------- |
| `yolov8_detection_v1`     | YOLOv8 model + metrics       |
| `resnet18_classification` | ResNet18 training & accuracy |
| `relog_yolov8n`           | Relogged model from Colab    |

---

## 📖 Feedback Loop (Bonus)

* Giao diện người dùng (`Streamlit`) cho phép chọn `track_id` và sửa nhãn
* Feedback được lưu vào `feedback.csv`
* Script retrain sẽ dùng các sample này để fine-tune lại mô hình phân loại

---

## 💼 Author

Nguyen Quang Trieu
MLOps Enthusiast | AI Learner | Senior Rigging Artist | Former Physics Teacher


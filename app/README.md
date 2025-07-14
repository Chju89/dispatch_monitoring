
### ✅ `app/README.md`

```markdown
# 🧠 app/: Modular Components for Realtime Object Monitoring

This folder contains the core logic for building a realtime object detection, tracking, and classification pipeline using:

- 🔍 YOLOv8 for object detection
- 🎯 DeepSORT for object tracking
- 🍽️ ResNet18 for object classification
- 🖍️ Utilities for drawing bounding boxes, labels, and track IDs

---

## 📁 Folder Structure

```

app/
├── detector.py         # YOLOv8 wrapper for object detection
├── tracker.py          # DeepSORT integration for object tracking
├── classifier.py       # ResNet18-based object classifier
├── utils/
│   └── draw\.py         # Visualization utilities (boxes, labels)

````

---

## 🔧 Dependencies

Make sure your environment includes:

- `ultralytics>=8.0`
- `torch`
- `opencv-python`
- `deep_sort_realtime`
- `numpy`

You can install them with:

```bash
pip install ultralytics deep_sort_realtime opencv-python torch
````

---

## 🧩 Module Overview

| Module          | Description                                              |
| --------------- | -------------------------------------------------------- |
| `detector.py`   | Loads YOLOv8 model and returns bounding boxes + conf     |
| `tracker.py`    | Wraps DeepSORT to maintain object identity across frames |
| `classifier.py` | Classifies each detected object (dish/tray type, etc.)   |
| `utils/draw.py` | Visualizes bbox, confidence, class name, and track ID    |

---

## 🚀 How to Use in a Script

Example usage (see `scripts/test_realtime_debug.py`):

```python
from app.detector import YoloDetector
from app.tracker import DeepSortTracker
from app.classifier import ResNetClassifier
from app.utils.draw import draw_detections

# Initialize models
detector = YoloDetector("models/detection/best.pt")
tracker = DeepSortTracker()
classifier = ResNetClassifier("models/classification/resnet18_dispatch.pt")

# Per frame logic
bboxes, confs = detector.predict(frame)
tracks = tracker.update_tracks(bboxes, confs, frame)

# For each track: get crop → classify → draw
```

---

## 📌 Notes

* Input/output format between modules is designed to be **clean and standardized**
* All models are loaded **once on init** to support **smooth realtime performance**
* Streamlit or OpenCV loop can integrate this logic easily

---

## 📮 Contributions

Feel free to extend or modify:

* Support more object classes
* Add action recognition
* Stream to web dashboard


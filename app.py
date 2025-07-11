import streamlit as st
import cv2
import numpy as np
import json
from pathlib import Path
from PIL import Image, ImageDraw

# ------------------------ Config ------------------------
VIDEO_PATH = Path("data/raw/video_shortened.mp4")
FEEDBACK_LOG_PATH = Path("data/feedback/")
object_types = ["dish", "tray"]
status_types = ["empty", "not_empty", "kakigori"]
bbox_colors = {"dish": "green", "tray": "red"}

# Load models
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
from torchvision import transforms
from torchvision.models import resnet18
import torch.nn as nn

# Detection - YOLO
yolo_model = YOLO("models/detection/best.pt")

# Tracking - DeepSORT
tracker = DeepSort(max_age=30)

# Classification - ResNet18
class ResNetClassifier:
    def __init__(self, model, id_to_class):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(), 
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.id_to_class = id_to_class

    def predict(self, image_crop):
        try:
            if image_crop is None or not isinstance(image_crop, np.ndarray):
                return 'unknown', 'unknown'

            if image_crop.ndim != 3 or image_crop.shape[2] != 3:
                return 'unknown', 'unknown'

            if image_crop.shape[0] < 5 or image_crop.shape[1] < 5:
                return 'unknown', 'unknown'

            img_tensor = self.transform(image_crop).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(img_tensor)
                pred = output.argmax(dim=1).item()

            label = self.id_to_class.get(pred, "unknown")
            if "_" not in label:
                return 'unknown', 'unknown'

            object_type, status_type = label.split("_", 1)
            return object_type, status_type

        except Exception:
            return 'unknown', 'unknown'

id_to_class = {
    0: 'dish_empty',
    1: 'dish_kakigori',
    2: 'dish_not_empty',
    3: 'tray_empty',
    4: 'tray_kakigori',
    5: 'tray_not_empty'
}

# Load full model (đã save bằng torch.save(model))
resnet_model = torch.load("models/classification/resnet18_dispatch.pt", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
# Khởi tạo class inference
classifier = ResNetClassifier(model=resnet_model, id_to_class=id_to_class)

# ------------------------ Helper ------------------------
def get_video_name():
    return VIDEO_PATH.stem

def get_feedback_log_name():
    return FEEDBACK_LOG_PATH / f"{get_video_name()}_feedback.json"

def read_frame_from_video(frame_idx):
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    success, frame = cap.read()
    cap.release()
    if not success:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def draw_bboxes(image, objects):
    img = Image.fromarray(image).copy()
    draw = ImageDraw.Draw(img)
    for obj in objects:
        x, y, w, h = obj["bbox"]
        color = bbox_colors.get(obj["object"], "blue")
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
        label = f"ID:{obj['id']} {obj['object']}_{obj['status']}"
        draw.text((x, y), label, fill=color)
    return img

def track_and_classify(frame_rgb, yolo_model, tracker, classifier):
    # 1. YOLOv8 detection
    results = yolo_model.predict(frame_rgb, imgsz=640, conf=0.005, verbose=False)[0]
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = yolo_model.model.names[cls_id]
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, label))

    # 2. DeepSORT tracking
    tracks = tracker.update_tracks(detections, frame=frame_rgb)

    # 3. Track + crop + classify
    h, w, _ = frame_rgb.shape
    objects = []

    crop = None
    bbox = []
    for track in tracks:
        # if not track.is_confirmed():
        #     continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())

        bbox.append((x1, y1, x2, y2))
        # Clamp to frame size
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        # Check valid bbox
        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        object_type, status_type = classifier.predict(crop)


        objects.append({
            "id": track_id,
            "object": object_type,
            "status": status_type,
            "bbox": [x1, y1, x2 - x1, y2 - y1]
        })

    return objects, (frame_rgb.shape)



def apply_feedback(objects, frame_name):
    log_path = get_feedback_log_name()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    new_entries = [{
        "object_id": obj["id"],
        "frame": frame_name,
        "new_object": obj["object"],
        "new_status": obj["status"],
        "bbox": obj["bbox"]
    } for obj in objects]

    # Load old log
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = []

    # Append if not exists
    appended = 0
    for entry in new_entries:
        if entry not in existing:
            existing.append(entry)
            appended += 1

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    if appended:
        st.success(f"✅ Appended {appended} new feedback entries to {log_path}")
    else:
        st.info("⚠️ No new feedback to save.")



# ------------------------ Streamlit UI ------------------------
st.set_page_config(layout="wide")
st.title("📦 Dispatch Monitoring - Realtime Inference")

# ------------------------ State init ------------------------
if "frame_idx" not in st.session_state:
    st.session_state.frame_idx = 0
if "deleted_ids" not in st.session_state:
    st.session_state.deleted_ids = set()
if "playing" not in st.session_state:
    st.session_state.playing = False
if "playback_speed" not in st.session_state:
    st.session_state.playback_speed = 1.0

# ------------------------ Frame control ------------------------
cap = cv2.VideoCapture(str(VIDEO_PATH))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

frame_idx = st.session_state.frame_idx
frame_rgb = read_frame_from_video(frame_idx)
if frame_rgb is None:
    st.error("❌ Cannot load frame.")
    st.stop()

# ------------------------ Inference ------------------------
objects, resault = track_and_classify(frame_rgb, yolo_model, tracker, classifier)

# Apply deletion filter
objects = [obj for obj in objects if obj["id"] not in st.session_state.deleted_ids]

st.success(resault)

# ------------------------ Layout -----------------------
# Layout 1 -2
col1, col2 = st.columns([3, 2])
with col1:
    st.image(draw_bboxes(frame_rgb, objects), caption=f"🖼️ Frame {frame_idx}", use_column_width=True)

with col2:
    st.markdown(f"### 🧾 Object List – Frame {frame_idx}")
    updated_objects = []

    for i, obj in enumerate(objects):
        cols = st.columns([1, 2, 2])

        with cols[0]:
            if st.button("🗑 Xoá", key=f"del_{obj['id']}"):
                st.session_state.deleted_ids.add(obj["id"])
                st.experimental_rerun()
            else:
                st.markdown(f"**ID {obj['id']}**")

        with cols[1]:
            default_obj = obj["object"] if obj["object"] in object_types else object_types[0]
            obj["object"] = st.selectbox("Object", object_types, index=object_types.index(default_obj), key=f"obj_{i}")

        with cols[2]:
            default_status = obj["status"] if obj["status"] in status_types else status_types[0]
            obj["status"] = st.selectbox("Status", status_types, index=status_types.index(default_status), key=f"status_{i}")

        updated_objects.append(obj)

    st.markdown("---")
    if st.button("✅ Apply Change"):
        apply_feedback(updated_objects, f"frame_{frame_idx:06d}.jpg")

        # Vẽ lại ảnh sau khi apply
        drawn = draw_bboxes(frame_rgb, objects)
        st.image(drawn, caption=f"🖼️ Updated Frame {frame_idx}", use_column_width=True)

# Layout 3 – Playback controls
st.markdown("---")
colA, colB, colC = st.columns([1, 1, 1])
with colA:
    if st.button("⬅️ Previous"):
        st.session_state.frame_idx = max(0, frame_idx - 1)
with colB:
    if st.button("▶️ Play/Pause"):
        st.session_state.playing = not st.session_state.playing
    st.selectbox("⏱ Playback Speed", options=[0.5, 1.0, 1.5, 2.0], key="playback_speed")
with colC:
    if st.button("➡️ Next"):
        st.session_state.frame_idx = min(total_frames - 1, frame_idx + 1)

if st.session_state.playing:
    import time
    time.sleep(max(0.05, 1.0 / st.session_state.playback_speed))
    st.session_state.frame_idx = min(total_frames - 1, frame_idx + 1)
    st.experimental_rerun()


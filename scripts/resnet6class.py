import streamlit as st
import torch
import cv2
import time
import numpy as np
from torchvision import transforms
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from PIL import Image
import json
from pathlib import Path
import tempfile

# --------------------- Setup ---------------------
st.set_page_config(layout="wide")
st.title("🏎️ Dispatch Monitoring - Realtime Inference")

object_types = ["dish", "tray"]
status_types = ["empty", "not_empty", "kakigori"]
bbox_colors = {"dish": "green", "tray": "red"}
color_map = {
    "green": (0, 255, 0),
    "red": (0, 0, 255),
}

label_map = {
    0: 'dish_empty',
    1: 'dish_kakigori',
    2: 'dish_not_empty',
    3: 'tray_empty',
    4: 'tray_kakigori',
    5: 'tray_not_empty'
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tray_model = YOLO("runs/detect/tray/yolov8n_aug_2_tray_only_1/weights/best.pt")
    dish_model = YOLO("runs/detect/dish/yolov8n_aug_2_dish_only_1/weights/best.pt")
    classifier = torch.load("models/classification/resnet18_dispatch.pt", map_location=device)
    classifier.eval()
    return tray_model, dish_model, classifier, device

@st.cache_resource
def init_tracker():
    return DeepSort(max_age=10)

@st.cache_resource
def get_video_capture(path):
    return cv2.VideoCapture(path)

video_file = st.file_uploader("🎮 Chọn video để phân tích", type=["mp4", "avi"])

if video_file:
    tmp_path = tempfile.NamedTemporaryFile(delete=False)
    tmp_path.write(video_file.read())
    VIDEO_PATH = tmp_path.name

    tray_model, dish_model, classifier, device = load_models()
    tracker = init_tracker()
    cap = get_video_capture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = 1.0 / fps if fps > 0 else 0.03

    if "is_playing" not in st.session_state:
        st.session_state.is_playing = False
    if "frame_idx" not in st.session_state:
        st.session_state.frame_idx = 0
    if "deleted_ids" not in st.session_state:
        st.session_state.deleted_ids = set()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    colA, colB, colC, colD = st.columns([1, 1, 1, 3])
    if colA.button("⏮️ Lùi lại"):
        st.session_state.frame_idx = max(0, st.session_state.frame_idx - 30)
        st.session_state.is_playing = False

    if colB.button("▶️ Play" if not st.session_state.is_playing else "⏸ Pause"):
        st.session_state.is_playing = not st.session_state.is_playing

    if colC.button("⏭️ Tiếp theo"):
        st.session_state.frame_idx = min(st.session_state.frame_idx + 1, total_frames - 1)
        st.session_state.is_playing = False

    cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_idx)
    ret, frame = cap.read()
    if not ret:
        st.stop()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame_rgb.shape
    resized = cv2.resize(frame_rgb, (w, h))

    tray_preds = tray_model.predict(source=resized, conf=0.05, iou=0.35, imgsz=1088, verbose=False)[0].boxes.data.tolist()
    dish_preds = dish_model.predict(source=resized, conf=0.05, iou=0.35, imgsz=1088, verbose=False)[0].boxes.data.tolist()
    boxes = tray_preds + dish_preds


    detections = []
    for box in boxes:
        x1, y1, x2, y2, conf, cls = map(float, box)
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        h, w, _ = resized.shape
        x1, y1, x2, y2 = map(lambda v, m: max(0, min(int(v), m - 1)), [x1, y1, x2, y2], [w, h, w, h])
        crop = resized[y1:y2, x1:x2]
        label = "unknown"

        try:
            pil_img = Image.fromarray(crop)
            input_tensor = transform(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = classifier(input_tensor)
                pred_cls = torch.argmax(output, dim=1).item()
                label = label_map.get(pred_cls, "unknown")
        except:
            pass
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, label))

    tracks = tracker.update_tracks(detections, frame=resized)
    objects = []
    for track in tracks:
        if not track.is_confirmed():
            continue
        tid = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        label = track.get_det_class()
        if tid in st.session_state.deleted_ids:
            continue
        parts = label.split("_")
        obj_type, status = parts if len(parts) == 2 else ("unknown", "unknown")
        objects.append({"id": f"{tid}_{st.session_state.frame_idx}", "bbox": [l, t, r, b], "object": obj_type, "status": status, "label": label})
        color = color_map.get(bbox_colors.get(obj_type, "green"))
        cv2.rectangle(resized, (l, t), (r, b), color, 2)
        cv2.putText(resized, f"ID {tid}: {label}", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    frame_placeholder = st.empty()
    with frame_placeholder.container():
        col1, col2 = st.columns([3, 2], gap="large")
        with col1:
            st.image(resized, channels="RGB", use_column_width=True)
        with col2:
            st.markdown(f"### 🎯 Danh sách Object - Frame: {st.session_state.frame_idx + 1}")
            updated_objects = []
            for i, obj in enumerate(objects):
                if obj["id"] in st.session_state.deleted_ids:
                    continue  # Skip deleted object
                st.markdown("---")
                cols = st.columns([1, 2, 2])
                with cols[0]:
                    if st.button("🗑 Xoá", key=f"del_{obj['id']}"):
                        st.session_state.deleted_ids.add(obj["id"])
                        continue
                    st.markdown(f"**ID {obj['id'].split('_')[0]}**")
                with cols[1]:
                    obj["object"] = st.selectbox("Object", object_types, index=object_types.index(obj["object"]) if obj["object"] in object_types else 0, key=f"obj_{obj['id']}")
                with cols[2]:
                    obj["status"] = st.selectbox("Status", status_types, index=status_types.index(obj["status"]) if obj["status"] in status_types else 0, key=f"status_{obj['id']}")
                updated_objects.append(obj)

            if st.button("✅ Apply Feedback", key=f"apply_feedback_btn_{st.session_state.frame_idx}"):
                frame_name = f"frame_{st.session_state.frame_idx}.jpg"
                log_path = Path(f"data/feedback/{video_file.name}_feedback.json")
                log_path.parent.mkdir(parents=True, exist_ok=True)

                try:
                    with open(log_path, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    existing = []

                new_entries = [{
                    "object_id": obj["id"],
                    "frame": frame_name,
                    "new_object": obj["object"],
                    "new_status": obj["status"],
                    "bbox": obj["bbox"]
                } for obj in updated_objects]

                appended = 0
                for entry in new_entries:
                    if entry not in existing:
                        existing.append(entry)
                        appended += 1

                with open(log_path, "w", encoding="utf-8") as f:
                    json.dump(existing, f, ensure_ascii=False, indent=2)

                if appended:
                    st.success(f"✅ Ghi {appended} feedback vào `{log_path.name}`")
                else:
                    st.info("⚠️ Không có thay đổi mới.")

    if st.session_state.is_playing:
        st.session_state.frame_idx += 1
        time.sleep(delay)
        st.experimental_rerun()


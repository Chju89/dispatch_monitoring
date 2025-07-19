import cv2
import streamlit as st
import torch
import torchvision.transforms as T
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from PIL import Image, ImageDraw
import time
import json
from pathlib import Path
import tempfile

# --------------------- Setup ---------------------
st.set_page_config(layout="wide")
st.title("📦 Dispatch Monitoring - Realtime Inference")

object_types = ["dish", "tray"]
status_types = ["empty", "not_empty", "kakigori"]
bbox_colors = {"dish": "green", "tray": "red"}

label_map = {
    0: 'dish_empty',
    1: 'dish_kakigori',
    2: 'dish_not_empty',
    3: 'tray_empty',
    4: 'tray_kakigori',
    5: 'tray_not_empty'
}

transform_cls = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@st.cache_resource
def load_models():
    # yolo = YOLO("yolov8s.pt")
    yolo = YOLO("models/detection/best.pt")
    resnet = torch.load("models/classification/resnet18_dispatch.pt", map_location="cpu")
    resnet.eval()
    return yolo, resnet

@st.cache_resource
def init_tracker():
    return DeepSort(max_age=15)

# --------------------- Video Upload ---------------------
video_file = st.file_uploader("🎬 Chọn video để phân tích", type=["mp4", "avi"])
if video_file:
    tmp_path = tempfile.NamedTemporaryFile(delete=False)
    tmp_path.write(video_file.read())
    VIDEO_PATH = Path(tmp_path.name)

    yolo_model, classifier = load_models()
    tracker = init_tracker()

    # --------------------- State init ---------------------
    if "frame_idx" not in st.session_state:
        st.session_state.frame_idx = 0
    if "deleted_ids" not in st.session_state:
        st.session_state.deleted_ids = set()
    if "playing" not in st.session_state:
        st.session_state.playing = False
    if "playback_speed" not in st.session_state:
        st.session_state.playback_speed = 1.0

    # --------------------- Read frame ---------------------
    def read_frame_from_video(frame_index):
        cap = cv2.VideoCapture(str(VIDEO_PATH))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def track_and_classify(frame, yolo_model, tracker, classifier):
        results = yolo_model.predict(source=frame, conf=0.1, iou=0.35, imgsz=640, verbose=False)
        boxes = results[0].boxes
        detections = []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            obj_crop = frame[y1:y2, x1:x2]
            try:
                img_pil = Image.fromarray(obj_crop)
                input_tensor = transform_cls(img_pil).unsqueeze(0)
                with torch.no_grad():
                    output = classifier(input_tensor)
                    cls_idx = torch.argmax(output, dim=1).item()
                    label = label_map[cls_idx]
            except:
                label = "unknown"
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, label))

        tracks = tracker.update_tracks(detections, frame=frame)
        objects = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            l, t, r, b = map(int, track.to_ltrb())
            obj = track.get_det_class()

            parts = obj.split("_")
            if len(parts) == 2:
                obj_type, status = parts
            else:
                obj_type, status = "unknown", "unknown"

            objects.append({
                "id": tid,
                "bbox": [l, t, r, b],
                "object": obj_type,
                "status": status
            })
        return objects

    def draw_bboxes(image, objects):
        img = Image.fromarray(image).copy()
        draw = ImageDraw.Draw(img)
        for obj in objects:
            if obj['object'] == "unknown":
                continue
            x, y, w, h = obj["bbox"]
            color = bbox_colors.get(obj["object"], "blue")
            draw.rectangle([x, y, w, h], outline=color, width=3)
            label = f"ID:{obj['id']} {obj['object']}_{obj['status']}"
            draw.text((x, y-20), label, fill=color, stroke_width=5, stroke_fill='white', spacing=5, bool=True)
        return img

    def apply_feedback(objects, frame_name):
        log_path = Path(f"data/feedback/{video_file.name}_feedback.json")
        log_path.parent.mkdir(parents=True, exist_ok=True)

        new_entries = [{
            "object_id": obj["id"],
            "frame": frame_name,
            "new_object": obj["object"],
            "new_status": obj["status"],
            "bbox": obj["bbox"]
        } for obj in objects]

        # Đọc log cũ an toàn
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing = []

        appended = 0
        for entry in new_entries:
            if entry not in existing:
                existing.append(entry)
                appended += 1

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

        if appended:
            st.success(f"✅ Đã ghi {appended} feedback → `{log_path.name}`")
        else:
            st.info("⚠️ Không có thay đổi mới.")



    cap = cv2.VideoCapture(str(VIDEO_PATH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # ------------------------ UI Layout ------------------------
    frame_idx = st.session_state.frame_idx
    frame_rgb = read_frame_from_video(frame_idx)
    if frame_rgb is None:
        st.error("❌ Không load được frame.")
        st.stop()
        
    objects = track_and_classify(frame_rgb, yolo_model, tracker, classifier)
    objects = [obj for obj in objects if obj["id"] not in st.session_state.deleted_ids]

    col1, col2 = st.columns([3, 2])
    with col1:
        st.image(draw_bboxes(frame_rgb, objects), caption=f"🖼️ Frame {frame_idx + 1} / {total_frames}", use_column_width=True)

    with col2:
        st.markdown(f"### 🧾 Object List – Frame {frame_idx + 1}")
        updated_objects = []

        for i, obj in enumerate(objects):
            st.markdown("---")
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

    # ------------------------ Playback Controls ------------------------
    st.markdown("---")
    colA, colB, colC = st.columns([1, 2, 1])
    with colA:
        if st.button("⬅️ Previous"):
            st.session_state.frame_idx = max(0, frame_idx - 1)

    with colB:
        play_label = "⏸ Pause" if st.session_state.playing else "▶️ Play"
        if st.button(play_label):
            st.session_state.playing = not st.session_state.playing
        st.selectbox("⏱ Tốc độ phát", options=[0.25, 0.5, 1.0, 1.5, 2.0], key="playback_speed")

    with colC:
        if st.button("➡️ Next"):
            st.session_state.frame_idx = min(total_frames - 1, frame_idx + 1)


    if st.session_state.playing:
        time.sleep(max(0.005, 1.0 / st.session_state.playback_speed))
        skip_n = 1 if st.session_state.playback_speed <= 1.0 else int(round(st.session_state.playback_speed))
        st.session_state.frame_idx = min(total_frames - 1, frame_idx + skip_n)
        st.experimental_rerun()




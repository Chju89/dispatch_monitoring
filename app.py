import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
import pickle
import json
from pathlib import Path

# ------------------------
# Config
# ------------------------
VIDEO_PATH = Path("data/raw/video_shortened.mp4")
UI_OBJECTS_PATH = Path("data/processed/tracking/logs/ui_objects.pkl")
#FEEDBACK_LOG_PATH = Path("data/feedback/feedback.pkl")
FEEDBACK_LOG_PATH = Path("data/feedback/")

object_types = ["dish", "tray"]
status_types = ["empty", "not_empty", "kakigori"]
bbox_colors = {"dish": "green", "tray": "red"}


st.set_page_config(layout="wide")

# ------------------------
# Utility functions
# ------------------------
def get_video_name():
    if not VIDEO_PATH.exists():
        return None
    return VIDEO_PATH.stem

def get_feedback_log_name():
    if not FEEDBACK_LOG_PATH.exists() or not VIDEO_PATH.exists():
        return None
    return FEEDBACK_LOG_PATH / f"{VIDEO_PATH.stem}_feedback.json"

def get_frame_list():
    if not UI_OBJECTS_PATH.exists():
        return []
    with open(UI_OBJECTS_PATH, "rb") as f:
        all_data = pickle.load(f)
    return [f"frame_{idx:06d}.jpg" for idx in sorted(all_data.keys())]

def load_objects_for_frame(frame_name):
    if not UI_OBJECTS_PATH.exists():
        return []
    with open(UI_OBJECTS_PATH, "rb") as f:
        all_data = pickle.load(f)
    try:
        frame_idx = int(Path(frame_name).stem.split("_")[-1])
    except:
        return []
    return all_data.get(frame_idx, [])

def read_frame_from_video(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    success, frame = cap.read()
    cap.release()
    if success:
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return None

def draw_bboxes(image, objects):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    for obj in objects:
        if obj.get("deleted"):color
        x, y, w, h = obj["bbox"]
        color = bbox_colors.get(obj["object"], "blue")
        draw.rectangle([x, y, x + w, y + h], outline=color, width=1)
        label = f"ID:{obj['id']} {obj['object']}_{obj['status']}"
        draw.text((x, y), label, align="top", fill=color, stroke_width=2, stroke_fill=color)
    return image

def apply_feedback(valid_objects, frame_name):
    feedback_log = []
    for obj in valid_objects:
        feedback_log.append({
            "object_id": obj["id"],
            "frame": frame_name,
            "new_object": obj["object"],
            "new_status": obj["status"],
            "bbox": obj["bbox"]
        })

    log_path = get_feedback_log_name()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Đọc file cũ nếu có
    if log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except json.JSONDecodeError:
            existing_data = []
    else:
        existing_data = []

    # Đảm bảo dữ liệu cũ là list
    if not isinstance(existing_data, list):
        existing_data = [existing_data]

    # Loại bỏ trùng lặp
    appended = 0
    for entry in feedback_log:
        if entry not in existing_data:
            existing_data.append(entry)
            appended += 1

    # Ghi lại toàn bộ nếu có dữ liệu mới
    if appended > 0:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
        st.success(f"✅ Appended {appended} new feedback entries to {log_path}")
    else:
        st.info(f"⚠️ No new feedback to append (all entries already exist)")

# ------------------------
# UI
# ------------------------

#st.set_page_config(layout="wide")
st.title("📦 Dispatch Feedback UI")

frame_list = get_frame_list()
if "frame_idx" not in st.session_state:
    st.session_state.frame_idx = 0
if "playing" not in st.session_state:
    st.session_state.playing = False
if "deleted_ids" not in st.session_state:
    st.session_state.deleted_ids = set()

if len(frame_list) == 0:
    st.warning("⚠️ Không có frame nào.")
    st.stop()

if "playback_speed" not in st.session_state:
    st.session_state.playback_speed = 1.0

frame_name = frame_list[st.session_state.frame_idx]
frame_idx = int(Path(frame_name).stem.split("_")[-1])
objects = load_objects_for_frame(frame_name)
objects = [obj for obj in objects if obj["id"] not in st.session_state.deleted_ids]
image = read_frame_from_video(VIDEO_PATH, frame_idx)

# Layout 1 + 2
col1, col2 = st.columns([3, 2], gap="large")

with col1:
    if image:
        drawn = draw_bboxes(image, objects)
        st.image(drawn, caption=f"🖼️ Frame {frame_idx}", use_column_width=True)
    else:
        st.error("❌ Không đọc được frame từ video.")

with col2:
    st.markdown(f"### 🧾 Danh sách object - Frame: {frame_idx}")
    valid_objects = []

    with st.container():
        for i, obj in enumerate(objects):
            st.markdown(f"**ID: {obj['id']} – BBox: {obj['bbox']}**")
            cols = st.columns([1, 2, 2])
            with cols[0]:
                if st.button("🗑 Xoá", key=f"del_{obj['id']}"):
                    st.session_state.deleted_ids.add(obj["id"])
                    st.experimental_rerun()
            with cols[1]:
                default_obj = obj["object"] if obj["object"] in object_types else object_types[0]
                obj["object"] = st.selectbox("Object", object_types, index=object_types.index(default_obj), key=f"obj_{i}")
            with cols[2]:
                default_status = obj["status"] if obj["status"] in status_types else status_types[0]
                obj["status"] = st.selectbox("Status", status_types, index=status_types.index(default_status), key=f"status_{i}")
            valid_objects.append(obj)

    st.markdown("---")
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        if st.button("✅ Apply Change"):
            apply_feedback(valid_objects, frame_name)

            # Vẽ lại ảnh sau khi apply
            drawn = draw_bboxes(image, valid_objects)
            st.image(drawn, caption=f"🖼️ Updated Frame {frame_idx}", use_column_width=True)

# Layout 3: điều khiển
st.markdown("---")
col_a, col_b, col_c = st.columns([1, 1, 1])
with col_a:
    if st.button("⬅️ Previous"):
        st.session_state.frame_idx = max(st.session_state.frame_idx - 1, 0)

with col_b:
    if st.button("▶️ Play/Pause"):
        st.session_state.playing = not st.session_state.playing

    st.selectbox(
        "⏱ Playback speed",
        options=[0.5, 1.0, 1.5, 2.0, 3.0],
        key="playback_speed"
    )


with col_c:
    if st.button("➡️ Next"):
        st.session_state.frame_idx = min(st.session_state.frame_idx + 1, len(frame_list) - 1)

if st.session_state.playing:
    import time
    speed = st.session_state.get("playback_speed", 1.0)
    time.sleep(max(0.05, 1.0 / speed))
    st.session_state.frame_idx = min(st.session_state.frame_idx + 1, len(frame_list) - 1)
    st.experimental_rerun()


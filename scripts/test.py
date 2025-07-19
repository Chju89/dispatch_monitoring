import streamlit as st
st.set_page_config(page_title="YOLOv8 + Classifier + Tracker", layout="centered")

import torch
import cv2
import time
import os
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
from deep_sort_realtime.deepsort_tracker import DeepSort
from PIL import Image

# ---------------- CONFIG ----------------
dish_model_path = "runs/detect/dish/yolov8n_aug_2_dish_only/weights/best.pt"
tray_model_path = "runs/detect/tray/yolov8n_aug_3_tray_only/weights/best.pt"
resnet_path = "models/classification/resnet18_dispatch.pt"
video_path = "data/raw/sample_video.mp4"

dish_conf_thresh = 0.1
dish_iou_thresh = 0.35
tray_conf_thresh = 0.1
tray_iou_thresh = 0.45
imgsz = 1088

# Class mapping (6-class)
label_map = {
    0: 'dish_empty',
    1: 'dish_kakigori',
    2: 'dish_not_empty',
    3: 'tray_empty',
    4: 'tray_kakigori',
    5: 'tray_not_empty'
}

colors = {
    "dish": (0, 255, 0),
    "tray": (255, 0, 0)
}

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dish_model = YOLO(dish_model_path)
    tray_model = YOLO(tray_model_path)

    classifier = torch.load(resnet_path, map_location=device)
    classifier.to(device)
    classifier.eval()

    return dish_model, tray_model, classifier, device

dish_model, tray_model, classifier, device = load_models()

# DeepSORT
tracker = DeepSort(max_age=30)

# Transform for ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

# ---------------- STREAMLIT UI ----------------
# st.set_page_config(page_title="YOLOv8 + Classifier + Tracker", layout="centered")
st.title("📹 Realtime Object Detection + Classification + Tracking")

# Init session state
for key in ["playing", "paused", "last_frame"]:
    if key not in st.session_state:
        st.session_state[key] = False if key != "last_frame" else None

col1, col2 = st.columns([1, 3])
with col1:
    play_button = st.button("▶️ Play")
    pause_button = st.button("⏸ Pause")

if play_button:
    st.session_state.playing = True
    st.session_state.paused = False
if pause_button:
    st.session_state.paused = True

frame_placeholder = st.empty()

# ---------------- MAIN LOOP ----------------
if st.session_state.playing:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = 1.0 / fps if fps > 0 else 0.04

    while cap.isOpened() and st.session_state.playing:
        if st.session_state.paused:
            if st.session_state.last_frame is not None:
                frame_placeholder.image(st.session_state.last_frame, channels="RGB", use_column_width=True)
            time.sleep(delay)
            continue

        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(frame, (imgsz, imgsz))

        # Inference
        dish_boxes = dish_model.predict(img_resized, conf=dish_conf_thresh, iou=dish_iou_thresh, verbose=False)[0].boxes
        tray_boxes = tray_model.predict(img_resized, conf=tray_conf_thresh, iou=tray_iou_thresh, verbose=False)[0].boxes

        all_detections = []

        for box_tensor, label in zip(dish_boxes.xyxy.cpu().numpy(), dish_boxes.cls.cpu().numpy()):
            all_detections.append((*box_tensor, float(dish_boxes.conf[0]), "dish"))

        for box_tensor, label in zip(tray_boxes.xyxy.cpu().numpy(), tray_boxes.cls.cpu().numpy()):
            all_detections.append((*box_tensor, float(tray_boxes.conf[0]), "tray"))

        # Format for DeepSORT
        detections = []
        for x1, y1, x2, y2, conf, cls_name in all_detections:
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_name))

        tracks = tracker.update_tracks(detections, frame=img_resized)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, w, h = track.to_ltrb()
            r, b = int(l + w), int(t + h)
            l, t = int(l), int(t)

            obj_crop = frame[t:b, l:r]
            if obj_crop.size == 0:
                continue

            # Classification
            img_pil = Image.fromarray(cv2.cvtColor(obj_crop, cv2.COLOR_BGR2RGB)).convert("RGB")
            input_tensor = transform(img_pil).unsqueeze(0).to(device) 
            with torch.no_grad():
                logits = classifier(input_tensor)
                pred_cls = torch.argmax(logits, dim=1).item()
                class_name = label_map.get(pred_cls, "unknown")

            # Draw box
            color = colors["dish"] if "dish" in class_name else colors["tray"]
            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            cv2.putText(frame, f"{class_name} #{track_id}", (l, t - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        frame_rgb_out = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.session_state.last_frame = frame_rgb_out
        frame_placeholder.image(frame_rgb_out, channels="RGB", use_column_width=True)

        time.sleep(delay)

    cap.release()
    st.session_state.playing = False


# app/main.py
import streamlit as st
import cv2
import torch
import numpy as np
from app.detector import YoloDetector
from app.tracker import ObjectTracker
from app.classifier import ObjectClassifier
from app.utils.draw import draw_boxes

st.set_page_config(layout="wide")
st.title("🎥 Realtime Dispatch Detection")

# Load models
DETECTION_MODEL = "models/detection/best.pt"
CLASSIFIER_MODEL = "models/classification/resnet18_dispatch.pt"

st.sidebar.header("Settings")
source = st.sidebar.selectbox("Select Camera", [0, "video.mp4"])

# Load components
detector = YoloDetector(DETECTION_MODEL)
tracker = ObjectTracker()
classifier = ObjectClassifier(CLASSIFIER_MODEL)

# Video stream
cap = cv2.VideoCapture(source)
stframe = st.empty()

id_to_label = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    bboxes, confidences = detector.predict(frame)
    tracks = tracker.update_tracks(bboxes, confidences, frame)

    for track in tracks:
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        crop = frame[y1:y2, x1:x2]

        if track_id not in id_to_label:
            label = classifier.predict(crop)
            id_to_label[track_id] = label

        label = id_to_label[track_id]
        frame = draw_boxes(frame, [(x1, y1, x2, y2, track_id, label)])

    stframe.image(frame, channels="BGR", use_column_width=True)

cap.release()

import cv2
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from pathlib import Path

# Paths
video_path = "data/raw/1473_CH05_20250501133703_154216.mp4"
output_dir = Path("data/processed/tracking")
output_dir.mkdir(parents=True, exist_ok=True)
output_video_path = output_dir / "tracked_output.mp4"

# Load YOLOv8 detection model
model = YOLO("runs/detect/yolov8n_dispatch_train_safe/weights/best.pt")

# Initialize DeepSort
tracker = DeepSort(max_age=30)

# Open video
cap = cv2.VideoCapture(video_path)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Video writer
out = cv2.VideoWriter(str(output_video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection
    results = model.predict(frame, verbose=False)[0]
    detections = []
    for box in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = box
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw tracks
    for track in tracks:
        if not track.is_confirmed():
            continue
        x, y, w_box, h_box = track.to_ltrb()
        track_id = track.track_id
        cv2.rectangle(frame, (int(x), int(y)), (int(w_box), int(h_box)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out.write(frame)

cap.release()
out.release()
print(f"Tracking completed. Output saved to: {output_video_path}")


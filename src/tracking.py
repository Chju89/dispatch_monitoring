import cv2
import os
from pathlib import Path
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

model = YOLO("models/detection/best.pt")
tracker = DeepSort(max_age=30)

input_dir = Path("data/raw/frames")
output_crop_dir = Path("data/processed/tracking/crops")
output_frame_dir = Path("data/processed/tracking/frames_with_id")
output_crop_dir.mkdir(parents=True, exist_ok=True)
output_frame_dir.mkdir(parents=True, exist_ok=True)

for frame_path in sorted(input_dir.glob("*.jpg")):
    frame = cv2.imread(str(frame_path))
    results = model.predict(source=frame, imgsz=640, conf=0.1, verbose=False)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        detections.append(([x1, y1, x2 - x1, y2 - y1], 0.9, cls_id))

    if not detections:
        print(f"[!] No detections in {frame_path.name}, skipping...")
        continue 

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cls_name = model.names[track.det_class] if hasattr(track, "det_class") else "object"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{cls_name}-{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        crop_dir = output_crop_dir / cls_name / f"{int(track_id):03d}"
        crop_dir.mkdir(parents=True, exist_ok=True)
        h, w, _ = frame.shape
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        crop_img = frame[y1:y2, x1:x2]
        if crop_img.size > 0:
            cv2.imwrite(str(crop_dir / frame_path.name), crop_img)

    cv2.imwrite(str(output_frame_dir / frame_path.name), frame)


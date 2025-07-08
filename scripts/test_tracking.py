import cv2
import os
from pathlib import Path
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLO model đã train
model = YOLO("models/detection/best.pt")
tracker = DeepSort(max_age=30)

# Đường dẫn input/output
input_dir = Path("data/raw/frames")
output_crop_dir = Path("data/processed/tracking/crops")
output_frame_dir = Path("data/processed/tracking/frames_with_id")
output_crop_dir.mkdir(parents=True, exist_ok=True)
output_frame_dir.mkdir(parents=True, exist_ok=True)

# Duyệt qua từng frame (tối đa 200)
for frame_path in sorted(input_dir.glob("*.jpg"))[:200]:
    frame = cv2.imread(str(frame_path))
    orig_h, orig_w = frame.shape[:2]

    # Resize về 640x640 để predict
    resized = cv2.resize(frame, (640, 640))
    results = model.predict(source=resized, imgsz=640, conf=0.1, verbose=False)[0]

    # Scale bbox về kích thước gốc
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1 = int(x1 * (orig_w / 640))
        y1 = int(y1 * (orig_h / 640))
        x2 = int(x2 * (orig_w / 640))
        y2 = int(y2 * (orig_h / 640))
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

        # Vẽ bounding box + track_id
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{cls_name}-{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Crop object và lưu
        crop_dir = output_crop_dir / cls_name / f"{int(track_id):03d}"
        crop_dir.mkdir(parents=True, exist_ok=True)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(orig_w, x2)
        y2 = min(orig_h, y2)
        crop_img = frame[y1:y2, x1:x2]
        if crop_img.size > 0:
            cv2.imwrite(str(crop_dir / frame_path.name), crop_img)

    # Lưu ảnh có bbox
    cv2.imwrite(str(output_frame_dir / frame_path.name), frame)


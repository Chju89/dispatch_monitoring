import os
import cv2
import torch
from pathlib import Path
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
from tqdm import tqdm

# === Cấu hình đường dẫn ===
frame_dir = Path("data/raw/frames")
output_dir = Path("data/processed/tracking")
frames_out = output_dir / "frames_with_id"
crops_out = output_dir / "crops"
logs_out = output_dir / "logs"

for p in [frames_out, crops_out, logs_out]:
    p.mkdir(parents=True, exist_ok=True)

# === Load model YOLO + DeepSORT ===
#model = YOLO("yolov8n.pt")
model = YOLO("runs/detect/yolov8n_dispatch_train_safe2/weights/best.pt")
tracker = DeepSort(max_age=30)

# === Load danh sách ảnh ===
frames = sorted(frame_dir.glob("*.jpg"))

log_file = open(logs_out / "tracks.txt", "w")

for frame_path in tqdm(frames, desc="Tracking"):
    frame = cv2.imread(str(frame_path))
    #results = model(frame)[0]
    results = model(frame, conf=0.1)[0]
    print(f"{frame_path.name} -> Detected: {len(results.boxes)} object(s)")


    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        class_name = model.names[cls]
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, class_name))

    if len(detections) > 0:
        tracks = tracker.update_tracks(detections, frame=frame)
    else:
        tracks = []

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb, class_name = track.to_ltrb(), track.det_class

        x1, y1, x2, y2 = map(int, ltrb)
        label = f"{class_name}-{track_id}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Lưu crop
        crop_dir = crops_out / class_name / f"{int(track_id):03d}"
        crop_dir.mkdir(parents=True, exist_ok=True)
        crop = frame[y1:y2, x1:x2]
        crop_name = crop_dir / frame_path.name
        if crop.size != 0:
            cv2.imwrite(str(crop_name), crop)

        # Ghi log
        log_file.write(f"{frame_path.name},{track_id},{class_name},{x1},{y1},{x2},{y2}\n")

    # Lưu frame đã gắn ID
    out_path = frames_out / frame_path.name
    cv2.imwrite(str(out_path), frame)

log_file.close()
print("✅ Done tracking with saved crops & logs.")


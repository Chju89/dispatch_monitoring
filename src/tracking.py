import cv2
import argparse
import pickle
from pathlib import Path
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def run_tracking(video_path):
    # Load YOLOv8 model
    model = YOLO("models/detection/best.pt")
    tracker = DeepSort(max_age=30)

    # Prepare output path
    output_log_path = Path("data/processed/tracking/logs/bbox_tracking_log.pkl")
    output_log_path.parent.mkdir(parents=True, exist_ok=True)

    bbox_log = {}

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model.predict(source=frame, imgsz=640, conf=0.1, verbose=False)[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], 0.9, cls_id))

        if not detections:
            frame_idx += 1
            continue

        tracks = tracker.update_tracks(detections, frame=frame)
        bbox_log[frame_idx] = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            cls_name = model.names[track.det_class] if hasattr(track, "det_class") else "object"

            bbox_log[frame_idx].append({
                "id": int(track_id),
                "object": cls_name,
                "bbox": [x1, y1, x2 - x1, y2 - y1]
            })

        frame_idx += 1

    cap.release()

    with open(output_log_path, "wb") as f:
        pickle.dump(bbox_log, f)

    print("✅ Tracking complete. Log saved to", output_log_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tracking on a video.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    args = parser.parse_args()

    run_tracking(args.video_path)
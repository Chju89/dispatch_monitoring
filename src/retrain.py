import json
import os
from pathlib import Path
import cv2

# ----------------------------
# Config
# ----------------------------
VIDEO_PATH = "data/raw/video_shortened.mp4"
FEEDBACK_FILE = Path("data/feedback") / f"feedback_{Path(VIDEO_PATH).stem}.json"
RETRAIN_DIR = Path("data/retrain")
CLS_SIZE = 224

CLASS_MAP_DET = {"dish": 0, "tray": 1}
CLASS_MAP_CLS = {
    "dish_empty": 0,
    "dish_kakigori": 1,
    "dish_not_empty": 2,
    "tray_empty": 3,
    "tray_kakigori": 4,
    "tray_not_empty": 5
}

with open(FEEDBACK_FILE, "r") as f:
    feedback = json.load(f)

det_img_dir = RETRAIN_DIR / "detection" / "images"
det_lbl_dir = RETRAIN_DIR / "detection" / "labels"
cls_img_dir = RETRAIN_DIR / "classifier"

for d in [det_img_dir, det_lbl_dir, cls_img_dir]:
    d.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
written_frames = set()

for item in feedback:
    frame_name = Path(item["frame"]).stem
    frame_idx = int(frame_name.split("_")[-1])
    object_id = item["object_id"]
    det_class = item["new_object"]
    cls_label = f"{item['new_object']}_{item['new_status']}"

    if det_class not in CLASS_MAP_DET or cls_label not in CLASS_MAP_CLS:
        continue

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    success, frame = cap.read()
    if not success:
        continue

    height, width = frame.shape[:2]
    x, y, w, h = map(int, item["bbox"])

    if frame_name not in written_frames:
        cv2.imwrite(str(det_img_dir / f"{frame_name}.jpg"), frame)
        written_frames.add(frame_name)

    cx = (x + w / 2) / width
    cy = (y + h / 2) / height
    nw = w / width
    nh = h / height
    with open(det_lbl_dir / f"{frame_name}.txt", "a") as f:
        f.write(f"{CLASS_MAP_DET[det_class]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    crop = frame[y:y+h, x:x+w]
    if crop.size == 0:
        continue
    resized = cv2.resize(crop, (CLS_SIZE, CLS_SIZE))
    out_dir = cls_img_dir / cls_label
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_name = f"{frame_name}_obj{object_id}.jpg"
    cv2.imwrite(str(out_dir / crop_name), resized)

cap.release()
print("✅ Retrain dataset created from JSON logs.")

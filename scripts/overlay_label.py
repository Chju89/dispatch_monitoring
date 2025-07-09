import cv2
import pandas as pd
from pathlib import Path

# Đường dẫn file và thư mục
summary_csv = Path("data/processed/tracking/summary.csv")
input_frames_dir = Path("data/processed/tracking/frames_with_id")
output_frames_dir = Path("data/processed/tracking/annotated_frames")
output_frames_dir.mkdir(parents=True, exist_ok=True)

# Đọc bảng mapping: track_id → predicted_label
summary_df = pd.read_csv(summary_csv)
track_id_to_label = dict(zip(summary_df["track_id"].astype(str), summary_df["predicted_label"]))

# Lặp qua tất cả các ảnh frame
for img_path in sorted(input_frames_dir.glob("*.jpg")):
    frame = cv2.imread(str(img_path))
    if frame is None:
        continue

    # Vẽ label cho mỗi track_id (nếu có)
    for track_id, label in track_id_to_label.items():
        # Text sẽ được vẽ gần ID
        text = f"ID:{track_id} {label}"

        # Tạm thời vẽ ở góc trái trên (nâng cấp sau nếu có bbox)
        cv2.putText(frame, text, (10, 30 + int(track_id) * 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Lưu frame đã annotate
    cv2.imwrite(str(output_frames_dir / img_path.name), frame)

print(f"[✅] Annotated frames saved to: {output_frames_dir}")


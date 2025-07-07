import cv2
import os

video_path = "data/raw/1473_CH05_20250501133703_154216.mp4"
output_dir = "data/raw/frames"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
count = 0
success = True

while success:
    success, frame = cap.read()
    if count % 30 == 0 and success:  # mỗi giây nếu video 30fps
        filename = os.path.join(output_dir, f"frame_{count:06d}.jpg")
        cv2.imwrite(filename, frame)
    count += 1

cap.release()
print("✅ Frame extraction done.")


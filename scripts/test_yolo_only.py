import cv2
from ultralytics import YOLO

# === Config ===
dish_model_path = "runs/detect/dish/yolov8n_aug_2_dish_only_1/weights/best.pt"
tray_model_path = "runs/detect/tray/yolov8n_aug_2_tray_only_1/weights/best.pt"
video_path = "data/raw/sample_video.mp4"
output_path = "data/raw/output_dish_tray_result.mp4"

tray_conf_thresh = 0.08
tray_iou_thresh = 0.45
dish_conf_thresh = 0.05
dish_iou_thresh = 0.35
imgsz = 1088

# === Load models ===
dish_model = YOLO(dish_model_path)
tray_model = YOLO(tray_model_path)

# === Load video ===
cap = cv2.VideoCapture(video_path)
w, h = int(cap.get(3)), int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

# === Output writer ===
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

# === Colors ===
color_dish = (0, 255, 0)   # Green
color_tray = (0, 0, 255)   # Red

# === Inference loop ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # === Inference ===
    dish_results = dish_model.predict(source=frame, conf=dish_conf_thresh, iou=dish_iou_thresh, imgsz=imgsz, verbose=False)
    tray_results = tray_model.predict(source=frame, conf=tray_conf_thresh, iou=tray_iou_thresh, imgsz=imgsz, verbose=False)

    # === Draw dish ===
    for box in dish_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_dish, 2)
        cv2.putText(frame, f'dish {conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_dish, 2)

    # === Draw tray ===
    for box in tray_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_tray, 2)
        cv2.putText(frame, f'tray {conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_tray, 2)

    # === Display and save ===
    cv2.imshow("YOLOv8 - dish + tray", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
out.release()
cv2.destroyAllWindows()


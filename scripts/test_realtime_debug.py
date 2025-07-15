import cv2
import torch
import torchvision.transforms as T
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from PIL import Image

# Load YOLOv8 pretrained
yolo_model = YOLO("models/detection/best.pt")  # dùng mô hình sẵn có từ ultralytics

# Load ResNet18 classification
resnet = torch.load("models/classification/resnet18_dispatch.pt", map_location="cpu")
resnet.eval()

label_map = {
    0: 'dish_empty',
    1: 'dish_kakigori',
    2: 'dish_not_empty',
    3: 'tray_empty',
    4: 'tray_kakigori',
    5: 'tray_not_empty'
}

# DeepSORT tracker
tracker = DeepSort(max_age=15)

# Transform đầu vào cho ResNet18
transform_cls = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load video
cap = cv2.VideoCapture("data/raw/sample_video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- YOLO DETECT ---
    results = yolo_model.predict(source=frame, conf=0.3, iou=0.5, verbose=False)
    boxes = results[0].boxes
    detections = []

    for box in boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        # Crop object để classify
        obj_crop = frame[y1:y2, x1:x2]
        try:
            img_pil = Image.fromarray(cv2.cvtColor(obj_crop, cv2.COLOR_BGR2RGB))
            input_tensor = transform_cls(img_pil).unsqueeze(0)  # (1, 3, 224, 224)

            with torch.no_grad():
                output = resnet(input_tensor)
                cls_idx = torch.argmax(output, dim=1).item()
                cls_name = label_map[cls_idx]
        except:
            cls_name = "unknown"

        # Append cho tracker
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_name))

    # --- TRACKING ---
    tracks = tracker.update_tracks(detections, frame=frame)

    # --- VẼ KẾT QUẢ ---
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, w, h = map(int, track.to_ltrb())
        label = track.get_det_class()  # tên class từ classify

        cv2.rectangle(frame, (l, t), (l + w, t + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} [ID {track_id}]", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)

    # --- HIỂN THỊ ---
    cv2.imshow("YOLO + Classify + Track", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


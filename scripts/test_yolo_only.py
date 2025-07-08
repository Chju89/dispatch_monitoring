import cv2
from ultralytics import YOLO
from pathlib import Path

# Load mô hình YOLO đã train
model = YOLO("models/detection/best.pt")
print(f"[INFO] Class names in model: {model.names}")

# Lấy ảnh test đầu tiên từ thư mục
img_path = sorted(Path("data/raw/frames").glob("*.jpg"))[10]
img_original = cv2.imread(str(img_path))

# Resize ảnh về kích thước đúng khi train
img_resized = cv2.resize(img_original, (640, 640))

# Dự đoán trên ảnh resized
results = model.predict(source=img_resized, imgsz=640, conf=0.05, verbose=False)[0]

# Tính hệ số scale ngược để đưa bbox về ảnh gốc
orig_h, orig_w = img_original.shape[:2]
scale_x = orig_w / 640
scale_y = orig_h / 640

classes_detected = set()
for box in results.boxes:
    x1, y1, x2, y2 = box.xyxy[0]
    cls_id = int(box.cls[0])
    cls_name = model.names[cls_id]
    classes_detected.add(cls_name)

    # Scale bbox về kích thước gốc
    x1 = int(x1 * scale_x)
    y1 = int(y1 * scale_y)
    x2 = int(x2 * scale_x)
    y2 = int(y2 * scale_y)

    # Vẽ lên ảnh gốc
    cv2.rectangle(img_original, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(img_original, cls_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

print(f"[INFO] Found {len(results.boxes)} boxes")
print(f"[INFO] Classes detected: {classes_detected}")

# Hiển thị ảnh kết quả
cv2.imshow("YOLO Detection (scaled)", img_original)
cv2.waitKey(0)
cv2.destroyAllWindows()


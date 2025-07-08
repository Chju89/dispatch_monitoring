import cv2
import matplotlib.pyplot as plt
import os

# ✅ Đường dẫn
img_dir = "data/raw/Dataset/Detection/val/images"
label_dir = "data/raw/Dataset/Detection/val/labels"
image_name = "img_000009.jpg"  # bạn có thể thay bằng file bất kỳ

# ✅ Load ảnh
img_path = os.path.join(img_dir, image_name)
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, _ = img.shape

# ✅ Load nhãn YOLOv8 từ file .txt
label_path = os.path.join(label_dir, image_name.replace(".jpg", ".txt"))
if not os.path.exists(label_path):
    print("⚠️ Không tìm thấy file nhãn.")
else:
    with open(label_path, "r") as f:
        for line in f.readlines():
            cls_id, x_center, y_center, bbox_w, bbox_h = map(float, line.strip().split())

            # Chuyển từ tỷ lệ YOLO (0-1) → pixel
            x1 = int((x_center - bbox_w / 2) * w)
            y1 = int((y_center - bbox_h / 2) * h)
            x2 = int((x_center + bbox_w / 2) * w)
            y2 = int((y_center + bbox_h / 2) * h)

            # Vẽ bbox + class
            color = (255, 0, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"class {int(cls_id)}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ✅ Hiển thị ảnh
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Kiểm tra bounding box từ nhãn")
    plt.show()


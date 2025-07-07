from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/yolov8n_dispatch_train_safe/weights/best.pt")

img = cv2.imread("data/raw/frames/frame_000030.jpg")  # hoặc 000060.jpg, v.v.
results = model(img, conf=0.1)
cv2.imshow("Detect", results[0].plot())
cv2.waitKey(0)
cv2.destroyAllWindows()


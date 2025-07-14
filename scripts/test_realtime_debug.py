# test_realtime_debug.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
from app.detector import YoloDetector
from app.tracker import ObjectTracker
from app.classifier import ObjectClassifier
from app.utils.draw import draw_boxes

VIDEO_IN = "data/raw/video_middle.mp4"

detector = YoloDetector("models/detection/best.pt")
tracker = ObjectTracker()
classifier = ObjectClassifier("models/classification/resnet18_dispatch.pt")

cap = cv2.VideoCapture(VIDEO_IN)
id_to_label = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    bboxes, confidences = detector.predict(frame)
    tracks = tracker.update_tracks(bboxes, confidences, frame)

    boxes = []
    for track in tracks:
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        crop = frame[y1:y2, x1:x2]

        if track_id not in id_to_label:
            label = classifier.predict(crop)
            id_to_label[track_id] = label

        label = id_to_label[track_id]
        boxes.append((x1, y1, x2, y2, track_id, label))

    frame = draw_boxes(frame, boxes)
    cv2.imshow("Realtime Pipeline", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# app/utils/draw.py
import cv2

def draw_boxes(frame, boxes):
    # boxes: list of (x1, y1, x2, y2, track_id, label)
    for x1, y1, x2, y2, track_id, label in boxes:
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID:{track_id} {label}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

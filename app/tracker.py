# app/tracker.py
from deep_sort_realtime.deepsort_tracker import DeepSort

class ObjectTracker:
    def __init__(self):
        self.tracker = DeepSort(max_age=30)

    def update_tracks(self, bboxes, confidences, frame):
        # bboxes: list of [x, y, w, h]
        # confidences: list of float
        tracks = self.tracker.update_tracks(bboxes, confidences, frame=frame)
        return [track for track in tracks if track.is_confirmed()]

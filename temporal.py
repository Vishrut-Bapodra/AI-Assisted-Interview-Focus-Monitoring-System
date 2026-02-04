from collections import deque
#import time
import numpy as np


class TemporalAnalyzer:
    def __init__(self, fps=10):
        self.fps = fps
        self.buffer = deque(maxlen=fps * 30)

        self.gaze_outside_start = None
        self.iris_invalid_start = None
        self.face_absent_start = None
        self.head_turned_start = None

    def update(self, signals):
        current_time = signals["timestamp"]
        self.buffer.append(signals)

        # -------- Gaze outside --------
        if signals["gaze_outside"]:
            if self.gaze_outside_start is None:
                self.gaze_outside_start = current_time
        else:
            self.gaze_outside_start = None

        # -------- Iris validity --------
        if not signals["iris_valid"]:
            if self.iris_invalid_start is None:
                self.iris_invalid_start = current_time
        else:
            self.iris_invalid_start = None

        # -------- Face detection --------
        if not signals["face_detected"]:
            if self.face_absent_start is None:
                self.face_absent_start = current_time
        else:
            self.face_absent_start = None

        # -------- Head turned --------
        if abs(signals["nose_offset_x"]) > 40:  # pixels; strict
            if self.head_turned_start is None:
                self.head_turned_start = current_time
        else:
            self.head_turned_start = None

        return self._compute_temporal_state(current_time)

    def _compute_temporal_state(self, current_time):
        gaze_outside_duration = (
            current_time - self.gaze_outside_start
            if self.gaze_outside_start else 0.0
        )

        iris_invalid_duration = (
            current_time - self.iris_invalid_start
            if self.iris_invalid_start else 0.0
        )

        face_absent_duration = (
            current_time - self.face_absent_start
            if self.face_absent_start else 0.0
        )

        head_turned_duration = (
            current_time - self.head_turned_start
            if self.head_turned_start else 0.0
        )

        nose_movements = [
            s["nose_movement"]
            for s in self.buffer
            if s["face_detected"]
        ]

        nose_variance = float(np.std(nose_movements)) if nose_movements else 0.0

        return {
            "gaze_outside_duration": round(gaze_outside_duration, 2),
            "iris_invalid_duration": round(iris_invalid_duration, 2),
            "face_absent_duration": round(face_absent_duration, 2),
            "head_turned_duration": round(head_turned_duration, 2),
            "nose_variance": round(nose_variance, 3)
        }

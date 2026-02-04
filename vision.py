import cv2
import numpy as np
import time
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class VisionProcessor:
    def __init__(self):
        base_options = python.BaseOptions(
            model_asset_path="models/face_landmarker.task"
        )

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )

        self.detector = vision.FaceLandmarker.create_from_options(options)

        self.RIGHT_EYE = [33, 133, 159, 145, 468]
        self.prev_nose = None

    def extract_signals(self, frame):
        h, w, _ = frame.shape

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        timestamp_ms = int(time.time() * 1000)
        result = self.detector.detect_for_video(mp_image, timestamp_ms)

        signals = {
            "face_detected": False,
            "iris_valid": False,
            "gaze_outside": False,
            "nose_movement": 0.0,
            "nose_offset_x": 0.0,
            "nose_offset_y": 0.0,
            "timestamp": time.time()
        }

        if not result.face_landmarks:
            self.prev_nose = None
            return signals

        landmarks = np.array([
            (lm.x * w, lm.y * h)
            for lm in result.face_landmarks[0]
        ])

        signals["face_detected"] = True

        # ---------------- Iris gaze logic ----------------
        try:
            hr, vr = self._calculate_gaze_ratio(landmarks)
            signals["iris_valid"] = True

            if hr < 0.35 or hr > 0.65 or vr < 0.35 or vr > 0.65:
                signals["gaze_outside"] = True

        except Exception:

            signals["iris_valid"] = False
            signals["gaze_outside"] = True

        # ---------------- Nose movement ----------------
        nose_x, nose_y = landmarks[1]

        if self.prev_nose is not None:
            dist = math.sqrt(
                (nose_x - self.prev_nose[0]) ** 2 +
                (nose_y - self.prev_nose[1]) ** 2
            )
            signals["nose_movement"] = dist

        self.prev_nose = (nose_x, nose_y)

        # ---------------- Nose offset ----------------
        xs = landmarks[:, 0]
        ys = landmarks[:, 1]

        face_center_x = (xs.min() + xs.max()) / 2
        face_center_y = (ys.min() + ys.max()) / 2

        signals["nose_offset_x"] = nose_x - face_center_x
        signals["nose_offset_y"] = nose_y - face_center_y

        return signals

    def _calculate_gaze_ratio(self, landmarks):
        left = landmarks[self.RIGHT_EYE[0]]
        right = landmarks[self.RIGHT_EYE[1]]
        top = landmarks[self.RIGHT_EYE[2]]
        bottom = landmarks[self.RIGHT_EYE[3]]
        iris = landmarks[self.RIGHT_EYE[4]]

        horizontal_ratio = (iris[0] - left[0]) / (right[0] - left[0] + 1)
        vertical_ratio = (iris[1] - top[1]) / (bottom[1] - top[1] + 1)

        return horizontal_ratio, vertical_ratio

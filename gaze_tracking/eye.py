import math
import numpy as np
import cv2
from .pupil import Pupil
from typing import Optional


class Eye(object):
    """
    This class creates a new frame to isolate the eye and
    initiates the pupil detection.
    """

    # MediaPipe face mesh eye landmark indices
    # Using proper eye contour points for accurate blinking detection
    # These are the standard EAR calculation points for MediaPipe
    LEFT_EYE_POINTS = [362, 385, 387, 263, 373, 380]  # Left eye outer contour
    RIGHT_EYE_POINTS = [33, 160, 158, 133, 153, 144]  # Right eye outer contour

    def __init__(self, original_frame: np.ndarray, landmarks, side: int, calibration) -> None:
        self.frame: Optional[np.ndarray] = None
        self.origin: Optional[tuple[int, int]] = None
        self.center: Optional[tuple[float, float]] = None
        self.pupil: Optional[Pupil] = None
        self.landmark_points: Optional[np.ndarray] = None

        self._analyze(original_frame, landmarks, side, calibration)

    @staticmethod
    def _middle_point(p1, p2) -> tuple[int, int]:
        """Returns the middle point (x,y) between two points

        Arguments:
            p1: First point (x, y)
            p2: Second point (x, y)
        """
        x = int((p1[0] + p2[0]) / 2)
        y = int((p1[1] + p2[1]) / 2)
        return (x, y)

    def _isolate(self, frame: np.ndarray, landmarks, points: list[int]) -> None:
        """Isolate an eye, to have a frame without other part of the face.

        Arguments:
            frame (numpy.ndarray): Frame containing the face
            landmarks: MediaPipe facial landmarks for the face region
            points (list): Points of an eye (from MediaPipe face mesh landmarks)
        """
        height, width = frame.shape[:2]
        
        # Convert MediaPipe normalized coordinates to pixel coordinates
        region = np.array([(int(landmarks.landmark[point].x * width), 
                           int(landmarks.landmark[point].y * height)) for point in points])
        region = region.astype(np.int32)
        self.landmark_points = region

        # Applying a mask to get only the eye
        black_frame = np.zeros((height, width), np.uint8)
        mask = np.full((height, width), 255, np.uint8)
        cv2.fillPoly(mask, [region], (0, 0, 0))
        eye = cv2.bitwise_not(black_frame, frame.copy(), mask=mask)

        # Cropping on the eye
        margin = 5
        min_x = np.min(region[:, 0]) - margin
        max_x = np.max(region[:, 0]) + margin
        min_y = np.min(region[:, 1]) - margin
        max_y = np.max(region[:, 1]) + margin

        self.frame = eye[min_y:max_y, min_x:max_x]
        self.origin = (min_x, min_y)

        height, width = self.frame.shape[:2]
        self.center = (width / 2, height / 2)

    def _blinking_ratio(self, landmarks, points: list[int], frame_shape: tuple) -> Optional[float]:
        """Calculates the Eye Aspect Ratio (EAR) that can indicate whether an eye is closed or not.
        EAR = (A + B + C) / (2.0 * D) where A, B, C are vertical distances and D is horizontal distance.

        Arguments:
            landmarks: MediaPipe facial landmarks for the face region
            points (list): Points of an eye (from MediaPipe face mesh landmarks)
            frame_shape: Shape of the original frame (height, width)

        Returns:
            The computed EAR ratio
        """
        height, width = frame_shape[:2]
        
        # Convert MediaPipe normalized coordinates to pixel coordinates
        # points[0], points[1] = left, right corners
        # points[2], points[3] = top points
        # points[4], points[5] = bottom points
        
        left = (int(landmarks.landmark[points[0]].x * width), int(landmarks.landmark[points[0]].y * height))
        right = (int(landmarks.landmark[points[1]].x * width), int(landmarks.landmark[points[1]].y * height))
        
        top1 = (int(landmarks.landmark[points[2]].x * width), int(landmarks.landmark[points[2]].y * height))
        top2 = (int(landmarks.landmark[points[3]].x * width), int(landmarks.landmark[points[3]].y * height))
        bottom1 = (int(landmarks.landmark[points[4]].x * width), int(landmarks.landmark[points[4]].y * height))
        bottom2 = (int(landmarks.landmark[points[5]].x * width), int(landmarks.landmark[points[5]].y * height))

        # Calculate vertical distances (A, B, C)
        A = math.hypot(top1[0] - left[0], top1[1] - left[1])
        B = math.hypot(top2[0] - right[0], top2[1] - right[1])
        C = math.hypot(bottom1[0] - bottom2[0], bottom1[1] - bottom2[1])
        
        # Calculate horizontal distance (D)
        D = math.hypot(left[0] - right[0], left[1] - right[1])

        try:
            # EAR = (A + B + C) / (2.0 * D)
            ear = (A + B + C) / (2.0 * D)
            return ear
        except ZeroDivisionError:
            return None

    def _analyze(self, original_frame: np.ndarray, landmarks, side: int, calibration) -> None:
        """Detects and isolates the eye in a new frame, sends data to the calibration
        and initializes Pupil object.

        Arguments:
            original_frame (numpy.ndarray): Frame passed by the user
            landmarks: MediaPipe facial landmarks for the face region
            side: Indicates whether it's the left eye (0) or the right eye (1)
            calibration (calibration.Calibration): Manages the binarization threshold value
        """
        if side == 0:
            points = self.LEFT_EYE_POINTS
        elif side == 1:
            points = self.RIGHT_EYE_POINTS
        else:
            return

        self._isolate(original_frame, landmarks, points)

        if not calibration.is_complete():
            calibration.evaluate(self.frame, side)

        threshold = calibration.threshold(side)
        self.pupil = Pupil(self.frame, threshold)
        
        # Set blinking to None since we're not using it
        self.blinking = None

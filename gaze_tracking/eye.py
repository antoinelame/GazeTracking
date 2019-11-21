import math
import numpy as np
import cv2
from .pupil import Pupil


# Detects the iris and estimates the position of the iris by calculating the centroid.
# Arguments:
#     eye_frame (numpy.ndarray): Frame containing an eye and nothing else
class Eye(object):

    LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
    RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]

    def __init__(self, original_frame, landmarks, side, calibration):
        self.frame = None
        self.origin = None
        self.center = None
        self.pupil = None

        self._analyze(original_frame, landmarks, side, calibration)

    # Returns the middle point (x,y) between two points
    # Arguments:
    #     p1 (dlib.point): First point
    #     p2 (dlib.point): Second point
    @staticmethod
    def _middle_point(p1, p2):
        x = int((p1.x + p2.x) / 2)
        y = int((p1.y + p2.y) / 2)
        return x, y

    # Isolate an eye, to have a frame without other part of the face.
    # Arguments:
    #     frame (numpy.ndarray): Frame containing the face
    #     landmarks (dlib.full_object_detection): Facial landmarks for the face region
    #     points (list): Points of an eye (from the 68 Multi-PIE landmarks)
    def _isolate(self, frame, landmarks, points):
        # put the six landmark coordinates for the eye into an array
        region = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in points])
        region = region.astype(np.int32)

        # Applying a mask to get only the eye
        height, width = frame.shape[:2]
        # black array the size of the webcam image
        black_frame = np.zeros((height, width), np.uint8)
        # white array the size of the webcam image
        mask = np.full((height, width), 255, np.uint8)
        # in the white mask, fill eye shape (contour given by landmarks) with black
        cv2.fillPoly(mask, [region], (0, 0, 0))
        # keep only the eye in the webcam image copy (par: src, dst, mask)
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

    # Calculates a ratio that can indicate whether an eye is closed or not.
    # It's the division of the width of the eye, by its height.
    # Arguments:
    #     landmarks (dlib.full_object_detection): Facial landmarks for the face region
    #     points (list): Points of an eye (from the 68 Multi-PIE landmarks)
    # Returns:
    #     The computed ratio
    def _blinking_ratio(self, landmarks, points):
        left = (landmarks.part(points[0]).x, landmarks.part(points[0]).y)
        right = (landmarks.part(points[3]).x, landmarks.part(points[3]).y)
        top = self._middle_point(landmarks.part(points[1]), landmarks.part(points[2]))
        bottom = self._middle_point(landmarks.part(points[5]), landmarks.part(points[4]))

        eye_width = math.hypot((left[0] - right[0]), (left[1] - right[1]))
        eye_height = math.hypot((top[0] - bottom[0]), (top[1] - bottom[1]))

        try:
            ratio = eye_width / eye_height
        except ZeroDivisionError:
            ratio = None

        return ratio

    # Detects and isolates the eye in a new frame, sends data to the calibration
    # and initializes Pupil object.
    # Arguments:
    #     original_frame (numpy.ndarray): Frame passed by the user
    #     landmarks (dlib.full_object_detection): Facial landmarks for the face region
    #     side: Indicates whether it's the left eye (0) or the right eye (1)
    #     calibration (calibration.Calibration): Manages the binarization threshold value
    def _analyze(self, original_frame, landmarks, side, calibration):
        if side == 0:
            points = self.LEFT_EYE_POINTS
        elif side == 1:
            points = self.RIGHT_EYE_POINTS
        else:
            return

        self.blinking = self._blinking_ratio(landmarks, points)
        self._isolate(original_frame, landmarks, points)

        if not calibration.is_complete():
            calibration.evaluate(self.frame, side)

        threshold = calibration.threshold(side)
        self.pupil = Pupil(self.frame, threshold)

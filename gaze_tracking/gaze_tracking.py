import cv2
from .eyes import EyesDetector
from .pupil import PupilDetector


class GazeTracking(object):
    """
    This class tracks the user's gaze.
    It provides useful information like the position of the eyes
    and the pupil and allows to know if the eyes are open or closed
    """

    def __init__(self):
        self.capture = cv2.VideoCapture(0)
        self.frame = None
        self.eyes = EyesDetector()
        self.pupil_left = PupilDetector()
        self.pupil_right = PupilDetector()

    def refresh(self):
        """Captures a new frame with the webcam and analyzes it."""
        _, self.frame = self.capture.read()
        self.eyes.process(self.frame)
        self.pupil_left.process(self.eyes.frame_left)
        self.pupil_right.process(self.eyes.frame_right)

    def pupil_left_coords(self):
        """Returns the coordinates of the left pupil"""
        try:
            x = self.eyes.frame_left_origin[0] + self.pupil_left.x
            y = self.eyes.frame_left_origin[1] + self.pupil_left.y
            return (x, y)
        except TypeError:
            return None

    def pupil_right_coords(self):
        """Returns the coordinates of the right pupil"""
        try:
            x = self.eyes.frame_right_origin[0] + self.pupil_right.x
            y = self.eyes.frame_right_origin[1] + self.pupil_right.y
            return (x, y)
        except TypeError:
            return None

    def horizontal_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        horizontal direction of the gaze. The extreme right is 0.0,
        the center is 0.5 and the extreme left is 1.0
        """
        try:
            pupil_right = self.pupil_right.x / (self.pupil_right.center[0] * 2 - 10)
            pupil_left = self.pupil_left.x / (self.pupil_left.center[0] * 2 - 10)
            return (pupil_right + pupil_left) / 2
        except TypeError:
            return None

    def vertical_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        vertical direction of the gaze. The extreme top is 0.0,
        the center is 0.5 and the extreme bottom is 1.0
        """
        try:
            pupil_right = self.pupil_right.y / (self.pupil_right.center[1] * 2 - 10)
            pupil_left = self.pupil_left.y / (self.pupil_left.center[1] * 2 - 10)
            return (pupil_right + pupil_left) / 2
        except TypeError:
            return None

    def is_right(self):
        """Returns true is the user is looking to the right"""
        try:
            return self.horizontal_ratio() <= 0.35
        except TypeError:
            return None

    def is_left(self):
        """Returns true is the user is looking to the left"""
        try:
            return self.horizontal_ratio() >= 0.65
        except TypeError:
            return None

    def is_center(self):
        """Returns true is the user is looking to the center"""
        return self.is_right() is not True and self.is_left() is not True

    def is_blinking(self):
        """Returns true if the user closes his eyes"""
        try:
            return self.eyes.blinking > 3.8
        except TypeError:
            return None

    def main_frame(self, highlighting=False):
        """Returns the main frame from the webcam

        Parameters:
            - highlighting (bool): Highlights pupils
        """
        frame = self.frame.copy()

        if highlighting:
            try:
                color = (0, 255, 0)
                x_left, y_left = self.pupil_left_coords()
                x_right, y_right = self.pupil_right_coords()
                cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
                cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
                cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
                cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)
            except TypeError:
                pass

        return frame

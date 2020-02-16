from __future__ import division
import os

import cv2
import dlib
from .eye import Eye


class GazeTracking(object):
    """
    This class tracks the user's gaze.
    It provides useful information like the position of the eyes
    and pupils and allows to know if the eyes are open or closed
    """
    def __init__(self, iris_calibration):

        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = iris_calibration

        # _face_detector is used to detect faces
        self._face_detector = dlib.get_frontal_face_detector()

        # _predictor is used to get facial landmarks of a given face
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "trained_models/shape_predictor_68_face_landmarks.dat"))
        self._predictor = dlib.shape_predictor(model_path)

    @property
    def pupils_located(self):
        """Checks that the pupils have been located"""
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False

    def _analyze(self):
        """Detects the face and initializes Eye objects"""
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector(frame)

        try:
            landmarks = self._predictor(frame, faces[0])
            self.eye_left = Eye(frame, landmarks, 0, self.calibration)
            self.eye_right = Eye(frame, landmarks, 1, self.calibration)

        except IndexError:
            self.eye_left = None
            self.eye_right = None

    def refresh(self, frame):
        """
        Refreshes the frame and analyzes it.

        :param: frame (numpy.ndarray) The frame to analyze
        :return: -
        """
        self.frame = frame
        self._analyze()

    def pupil_left_coords(self):
        """Returns the coordinates of the left pupil"""
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return x, y

    def pupil_right_coords(self):
        """Returns the coordinates of the right pupil"""
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return x, y

    def horizontal_ratio(self):
        """
        :return: a number between 0.0 and 1.0 that indicates the direction of the gaze.
        The extreme right is 0.0, the center is 0.5 and the extreme left is 1.0 The actual
        min and max that can be achieved by the human eye lies around 0.2 and 0.8 Eye
        and pupil coordinates are measured from edge of eye-frame, with 5 px padding
        all around.
        """
        if self.pupils_located:
            # Remove padding (5 px) from both the eye coord and pupil coord to get
            # the two actual distances from eye corner. Then divide to obtain relative position
            # of pupil wrt eye width.
            pupil_left = self.eye_left.pupil.x / (2 * (self.eye_left.center[0] - 5))
            pupil_right = self.eye_right.pupil.x / (2 * (self.eye_right.center[0] - 5))
            return (pupil_left + pupil_right) / 2

    def vertical_ratio(self):
        """
        :return: a number between 0.0 and 1.0 that indicates the
        vertical direction of the gaze. The extreme top is 0.0,
        the center is 0.5 and the extreme bottom is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.y / (2 * (self.eye_left.center[1] - 5))
            pupil_right = self.eye_right.pupil.y / (2 * (self.eye_right.center[1] - 5))
            return (pupil_left + pupil_right) / 2

    def is_right(self):
        """Returns true if the user is looking to the right"""
        if self.pupils_located:
            return self.horizontal_ratio() <= 0.35

    def is_left(self):
        """Returns true if the user is looking to the left"""
        if self.pupils_located:
            return self.horizontal_ratio() >= 0.65

    def is_center(self):
        """Returns true if the user is looking to the center"""
        if self.pupils_located:
            return self.is_right() is not True and self.is_left() is not True

    def is_up(self):
        """Returns true if the user is looking to the right"""
        if self.pupils_located:
            return self.vertical_ratio() <= 0.35

    def is_down(self):
        """Returns true if the user is looking to the left"""
        if self.pupils_located:
            return self.vertical_ratio() >= 0.65

    def is_level(self):
        """Returns true if the user is looking to the center"""
        if self.pupils_located:
            return self.is_up() is not True and self.is_down() is not True

    def is_blinking(self):
        """Returns true if the user closes his eyes. Is based on eye_width / eye_height"""
        if self.pupils_located:
            blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2
            return blinking_ratio > 4.5  # 3.8 original val

    def annotated_frame(self):
        """Returns the main frame with pupils marked with a green cross"""
        frame = self.frame.copy()

        if self.pupils_located:
            color = (0, 255, 0)
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()
            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)

        return frame

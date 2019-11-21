from __future__ import division
import math
import os
import cv2
import dlib
from .eye import Eye
from .gazecalibration import GazeCalibration


# This class tracks the user's gaze.
# It provides useful information like the position of the eyes
# and pupils and allows to know if the eyes are open or closed
class GazeTracking(object):

    def __init__(self, iris_calibration):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = iris_calibration
        self.current_iris_size = None

        # _face_detector is used to detect faces
        self._face_detector = dlib.get_frontal_face_detector()

        # _predictor is used to get facial landmarks of a given face
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "trained_models/shape_predictor_68_face_landmarks.dat"))
        self._predictor = dlib.shape_predictor(model_path)

    # Check that the pupils have been located
    @property
    def pupils_located(self):
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False

    # Detects the face and initialize Eye objects
    def _analyze(self):
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector(frame)

        try:
            landmarks = self._predictor(frame, faces[0])
            self.eye_left = Eye(frame, landmarks, 0, self.calibration)
            self.eye_right = Eye(frame, landmarks, 1, self.calibration)

        except IndexError:
            self.eye_left = None
            self.eye_right = None

    # Refreshes the frame and analyzes it.
    # Arguments:
    #    frame (numpy.ndarray): The frame to analyze
    def refresh(self, frame):
        self.frame = frame
        self._analyze()

    # Returns the coordinates of the left pupil
    def pupil_left_coords(self):
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return x, y

    # Returns the coordinates of the right pupil
    def pupil_right_coords(self):
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return x, y

    # Returns a number between 0.0 and 1.0 that indicates the direction of the gaze. The extreme right is 0.0,
    # the center is 0.5 and the extreme left is 1.0 The actual min and max that can be achieved by the
    # human eye lies around 0.2 and 0.8 Eye and pupil coordinates are measured from edge of eye-frame,
    # with 5 px padding all around.
    def horizontal_ratio(self):
        if self.pupils_located:
            # Remove padding (5 px) from both the eye coord and pupil coord to get
            # the two actual distances from eye corner. Then divide to obtain relative position
            # of pupil wrt eye width.
            pupil_left = self.eye_left.pupil.x / (2 * (self.eye_left.center[0] - 5))
            pupil_right = self.eye_right.pupil.x / (2 * (self.eye_right.center[0] - 5))
            return (pupil_left + pupil_right) / 2

    # Returns a number between 0.0 and 1.0 that indicates the
    # vertical direction of the gaze. The extreme top is 0.0,
    # the center is 0.5 and the extreme bottom is 1.0
    def vertical_ratio(self):
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.y / (2 * (self.eye_left.center[1] - 5))
            pupil_right = self.eye_right.pupil.y / (2 * (self.eye_right.center[1] - 5))
            return (pupil_left + pupil_right) / 2

    # Estimate the point of gaze on the computer screen based on the
    # horizontal and vertical ratios.
    def point_of_gaze(self, gaze_calib: GazeCalibration):
        if self.current_iris_size is None:
            self.current_iris_size = gaze_calib.base_iris_size
        # if > 1, curr is further away compared to base, if < 1 user has moved nearer
        dist_factor = gaze_calib.base_iris_size / self.current_iris_size
        if self.pupils_located:
            est_x = round((max(gaze_calib.leftmost_hr - self.horizontal_ratio(), 0) * gaze_calib.fsw * dist_factor) /
                          (gaze_calib.leftmost_hr - gaze_calib.rightmost_hr))
            est_y = round((max(self.vertical_ratio() - gaze_calib.top_vr, 0) * gaze_calib.fsh * dist_factor) /
                          (gaze_calib.bottom_vr - gaze_calib.top_vr))
            if self.looking_straight_ahead(est_x, est_y, gaze_calib):
                self.current_iris_size = self.iris_diameter()
            return est_x, est_y

    @staticmethod
    def looking_straight_ahead(est_x, est_y, gaze_calib):
        wmargin = gaze_calib.fsw * 0.2
        hmargin = gaze_calib.fsh * 0.3
        wmiddle = gaze_calib.fsw / 2
        hmiddle = gaze_calib.fsh / 2
        return wmiddle - wmargin < est_x < wmiddle + wmargin and hmiddle - hmargin < est_y < hmiddle + hmargin

    # Returns the absolute size (number of pixels) that the iris takes up on
    # the surface of the eye frame. frame (numpy.ndarray): Binarized iris frame,
    # i.e. eye-sized frame, where only the iris is visible (is black).
    def iris_diameter(self):
        right_frame = self.eye_right.frame[5:-5, 5:-5]
        left_frame = self.eye_left.frame[5:-5, 5:-5]
        nb_blacks_r = cv2.countNonZero(right_frame)
        nb_blacks_l = cv2.countNonZero(left_frame)
        # nb_blacks: approximation for iris area
        rad_r = math.sqrt(nb_blacks_r / math.pi)
        rad_l = math.sqrt(nb_blacks_l / math.pi)
        # return diameter (avg over right and left eye)
        return rad_r + rad_l

    # Returns true if the user is looking to the right
    def is_right(self):
        if self.pupils_located:
            return self.horizontal_ratio() <= 0.35

    # Returns true if the user is looking to the left
    def is_left(self):
        if self.pupils_located:
            return self.horizontal_ratio() >= 0.65

    # Returns true if the user is looking to the center
    def is_center(self):
        if self.pupils_located:
            return self.is_right() is not True and self.is_left() is not True

    # Returns true if the user is looking to the right
    def is_up(self):
        if self.pupils_located:
            return self.vertical_ratio() <= 0.35

    # Returns true if the user is looking to the left
    def is_down(self):
        if self.pupils_located:
            return self.vertical_ratio() >= 0.65

    # Returns true if the user is looking to the center
    def is_level(self):
        if self.pupils_located:
            return self.is_up() is not True and self.is_down() is not True

    # Returns true if the user closes his eyes
    # Is based on eye_width / eye_height
    def is_blinking(self):
        if self.pupils_located:
            blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2
            return blinking_ratio > 4.5  # 3.8 original val

    # Returns the main frame with pupils highlighted
    def annotated_frame(self):
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

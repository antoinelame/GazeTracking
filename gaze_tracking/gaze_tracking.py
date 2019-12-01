from __future__ import division
import math
import os
from collections import deque

import cv2
import dlib
from .eye import Eye
from .gazecalibration import GazeCalibration


class GazeTracking(object):
    """
    This class tracks the user's gaze.
    It provides useful information like the position of the eyes
    and pupils and allows to know if the eyes are open or closed
    """
    def __init__(self, iris_calibration, monitor, stabilize):
        self.stabilize = stabilize
        self.gaze_cluster_size = 2  # three gaze points in a cluster
        self.mx_intra_cluster_variation_const = 0.3  # max variation within a cluster = 30 % of screen size

        self.mx_intra_cluster_dist = math.sqrt((self.mx_intra_cluster_variation_const * monitor.width) ** 2
                                               + (self.mx_intra_cluster_variation_const * monitor.height) ** 2)

        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = iris_calibration
        self.current_iris_size = None
        self.current_cluster = deque({}, self.gaze_cluster_size)  # max_size, when queue is full
        self.candidate_cluster = deque({}, self.gaze_cluster_size)

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

    def point_of_gaze(self, gaze_calib: GazeCalibration):
        """
        Estimate the point of gaze on the computer screen based on the
        horizontal and vertical ratios.

        :param gaze_calib: a GazeCalibration object, which holds an baseline iris_size,
        and the extreme values for gaze_ratios (for each edge of the computer screen)
        :return: estimated gaze x, y coordinates on the computer screen if stabilization is off
        alternatively a stabilized (clustered) coordinates, if stabilization is on
        """
        if self.current_iris_size is None:
            self.current_iris_size = gaze_calib.base_iris_size
        # if > 1, curr is further away compared to base, if < 1 user has moved nearer
        dist_factor = gaze_calib.base_iris_size / self.current_iris_size
        if self.pupils_located:
            est_x = round((max(gaze_calib.leftmost_hr - self.horizontal_ratio(), 0) * gaze_calib.fsw * dist_factor) /
                          (gaze_calib.leftmost_hr - gaze_calib.rightmost_hr))
            est_y = round((max(self.vertical_ratio() - gaze_calib.top_vr, 0) * gaze_calib.fsh * dist_factor) /
                          (gaze_calib.bottom_vr - gaze_calib.top_vr))
            if self.stabilize:
                stab_x, stab_y = self.stabilized(est_x, est_y)
                if self.looking_straight_ahead(stab_x, stab_y, gaze_calib):
                    self.current_iris_size = self.measure_iris_diameter()
                return stab_x, stab_y
            else:
                return est_x, est_y
        else:
            return None, None

    def stabilized(self, x, y):
        """
        Stabilizes an estimated point of gaze (EPOG) by comparing it to previous EPOGs.

        :param x: estimated screen coordinate x (horizontal from left)
        :param y: estimated screen y (vertical from top)
        :return: x, y if this point is within r distance to k previously estimated gaze points
        otherwise: return the middle-point of the previous k estimates
        """
        k = self.gaze_cluster_size
        r = self.mx_intra_cluster_dist
        # bootstrap the first cluster
        if self.current_cluster.__len__() < k:
            self.current_cluster.append((x, y))
            return x, y
        # candidate cluster = candidate for possible eye movement coord
        # if candidate cluster is non-empty, then
        # check if dist to each member in the candidate cluster < r, then
        # include the new point and shift this cluster one time step (remove oldest)
        if self.candidate_cluster.__len__() > 0:
            if self._within_cluster(self.candidate_cluster, (x, y), r):
                self.candidate_cluster.append((x, y))
                # if the candidate cluster is big enough (size = k), then swap candidate to current
                if self.candidate_cluster.__len__() == k:
                    self.current_cluster = self.candidate_cluster
                    self.candidate_cluster.clear()
                return self.centroid(self.candidate_cluster, x, y)
            else:
                self.candidate_cluster.clear()
        # if max dist to each member in the current cluster < r, then
        # include the new point and shift this cluster one time step (remove oldest)
        if self._within_cluster(self.current_cluster, (x, y), r):
            self.current_cluster.append((x, y))
        else:
            self.candidate_cluster.append((x, y))
        return self.centroid(self.current_cluster, x, y)

    @staticmethod
    def centroid(cluster: deque, x, y):
        if cluster.__len__() > 0:
            xs = 0
            ys = 0
            for pc in cluster:
                xs = xs + pc[0]
                ys = ys + pc[1]
            return xs / cluster.__len__(), ys / cluster.__len__()
        else:
            return x, y

    @staticmethod
    def _within_cluster(c, p, r):
        """
        Check if point p is within r distance to elements in cluster c

        :param c (collections.deque): the cluster
        :param p: a point (x, y)
        :param r: max distance allowed from p to any element in the cluster
        :return: Per design: returns True even when cluster is empty
        """
        for pc in c:
            dist = math.sqrt((pc[0] - p[0]) ** 2 + (pc[1] - p[1]) ** 2)
            if dist > r:
                return False
        return True

    @staticmethod
    def looking_straight_ahead(est_x, est_y, gaze_calib):
        """
        :param est_x: EPOG x coordinate
        :param est_y: EPOG y coordinate
        :param gaze_calib: (GazeCalibration) calibration object holding the screen size
        :return: True if EPOG is in the center region of the screen
        """
        wmargin = gaze_calib.fsw * 0.3
        hmargin = gaze_calib.fsh * 0.5
        wmiddle = gaze_calib.fsw / 2
        hmiddle = gaze_calib.fsh / 2
        return wmiddle - wmargin < est_x < wmiddle + wmargin and hmiddle - hmargin < est_y < hmiddle + hmargin

    def measure_iris_diameter(self):
        """
        :return: Returns the iris diameter based on the absolute size (number of pixels)
        that the iris takes up on the surface of the eye frame.
        """
        #  frame: (numpy.ndarray) Binarized iris frame, i.e. eye-sized frame,
        #  where only the iris is visible (is black).
        right_frame = self.eye_right.frame[5:-5, 5:-5]
        left_frame = self.eye_left.frame[5:-5, 5:-5]
        nb_blacks_r = cv2.countNonZero(right_frame)
        nb_blacks_l = cv2.countNonZero(left_frame)
        # nb_blacks: approximation for iris area
        rad_r = math.sqrt(nb_blacks_r / math.pi)
        rad_l = math.sqrt(nb_blacks_l / math.pi)
        # return diameter (avg over right and left eye)
        return rad_r + rad_l

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

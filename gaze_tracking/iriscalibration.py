from __future__ import division
import cv2
from .pupil import Pupil


class IrisCalibration(object):
    """
    This class calibrates the pupil detection algorithm by finding the
    best binarization threshold value for the person and the webcam.
    """

    def __init__(self):
        self.nb_frames = 20
        self.thresholds_left = []
        self.thresholds_right = []

    def is_complete(self):
        """Returns true if the calibration is completed"""
        return len(self.thresholds_left) >= self.nb_frames and len(self.thresholds_right) >= self.nb_frames

    def threshold(self, side):
        """
        Returns the threshold value for the given eye.

        :param side: Indicates whether it's the left eye (0) or the right eye (1)
        :return: Threshold for the given eye (left or right)
        """
        if side == 0:
            return int(sum(self.thresholds_left) / len(self.thresholds_left))
        elif side == 1:
            return int(sum(self.thresholds_right) / len(self.thresholds_right))

    @staticmethod
    def iris_size(frame):
        """
        Returns the percentage of space that the iris takes up on
        the surface of the eye frame.

        :param frame: (numpy.ndarray) Binarized iris frame
        :return: The relative size (number of pixels) of the iris w.r.t. the eye
        """
        frame = frame[5:-5, 5:-5]
        height, width = frame.shape[:2]
        nb_pixels = height * width
        nb_blacks = nb_pixels - cv2.countNonZero(frame)
        return nb_blacks / nb_pixels

    @staticmethod
    def find_best_threshold(eye_frame):
        """
        Calculates the optimal threshold to binarize the frame for the given eye.

        :param eye_frame: (numpy.ndarray) Frame of the eye to be analyzed
        :return: Best threshold (that gives an iris with 48% occupancy of the eye)
        """
        average_iris_size = 0.48
        trials = {}

        for threshold in range(0, 200, 1):
            iris_frame = Pupil.image_processing(eye_frame, threshold)
            trials[threshold] = IrisCalibration.iris_size(iris_frame)

        best_threshold, iris_size = min(trials.items(), key=(lambda p: abs(p[1] - average_iris_size)))
        return best_threshold

    def evaluate(self, eye_frame, side):
        """
        Improves calibration by taking into consideration the given image.

        :param eye_frame: (numpy.ndarray) Frame of the eye
        :param side: Indicates whether it's the left eye (0) or the right eye (1)
        :return: -
        """
        threshold = self.find_best_threshold(eye_frame)

        if side == 0:
            self.thresholds_left.append(threshold)
        elif side == 1:
            self.thresholds_right.append(threshold)

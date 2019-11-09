from __future__ import division
import cv2
from .pupil import Pupil


#     This class calibrates the pupil detection algorithm by finding the
#     best binarization threshold value for the person and the webcam.
class Calibration(object):

    def __init__(self):
        self.nb_frames = 20
        self.thresholds_left = []
        self.thresholds_right = []

    # Returns true if the calibration is completed
    def is_complete(self):
        return len(self.thresholds_left) >= self.nb_frames and len(self.thresholds_right) >= self.nb_frames

    # Returns the threshold value for the given eye.
    # Argument:
    #    side: Indicates whether it's the left eye (0) or the right eye (1)
    def threshold(self, side):
        if side == 0:
            return int(sum(self.thresholds_left) / len(self.thresholds_left))
        elif side == 1:
            return int(sum(self.thresholds_right) / len(self.thresholds_right))

    # Returns the percentage of space that the iris takes up on
    # the surface of the eye.
    # Argument:
    #     frame (numpy.ndarray): Binarized iris frame
    @staticmethod
    def iris_size(frame):
        frame = frame[5:-5, 5:-5]
        height, width = frame.shape[:2]
        nb_pixels = height * width
        nb_blacks = nb_pixels - cv2.countNonZero(frame)
        return nb_blacks / nb_pixels

    # Calculates the optimal threshold to binarize the
    # frame for the given eye.
    # Argument:
    #     eye_frame (numpy.ndarray): Frame of the eye to be analyzed
    @staticmethod
    def find_best_threshold(eye_frame):
        average_iris_size = 0.48
        trials = {}

        for threshold in range(5, 100, 5):
            iris_frame = Pupil.image_processing(eye_frame, threshold)
            trials[threshold] = Calibration.iris_size(iris_frame)

        best_threshold, iris_size = min(trials.items(), key=(lambda p: abs(p[1] - average_iris_size)))
        return best_threshold

    # Improves calibration by taking into consideration the
    # given image.
    # Arguments:
    #     eye_frame (numpy.ndarray): Frame of the eye
    #     side: Indicates whether it's the left eye (0) or the right eye (1)
    def evaluate(self, eye_frame, side):
        threshold = self.find_best_threshold(eye_frame)

        if side == 0:
            self.thresholds_left.append(threshold)
        elif side == 1:
            self.thresholds_right.append(threshold)

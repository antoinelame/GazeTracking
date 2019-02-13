import numpy as np
import cv2


class PupilDetector(object):
    """
    This class detects the iris of an eye and estimates
    the position of the pupil
    """

    def __init__(self):
        self.modified_frame = None
        self.center = None
        self.x = None
        self.y = None

    @staticmethod
    def image_processing(eye_frame):
        """Performs operations on the eye frame to isolate the iris

        Arguments:
            eye_frame (numpy.ndarray): Frame containing an eye and nothing else

        Returns:
            A frame with a single element representing the iris
        """
        kernel = np.ones((3, 3), np.uint8)
        new_frame = cv2.bilateralFilter(eye_frame, 10, 15, 15)
        new_frame = cv2.threshold(new_frame, 20, 255, cv2.THRESH_BINARY)[1]
        new_frame = cv2.erode(new_frame, kernel, iterations=3)
        new_frame = cv2.dilate(new_frame, kernel, iterations=2)
        return new_frame

    def process(self, frame):
        """Run iris detection and pupil estimation"""
        if frame is None:
            return

        self.modified_frame = self.image_processing(frame)

        height, width = self.modified_frame.shape[:2]
        self.center = (width / 2, height / 2)

        _, contours, _ = cv2.findContours(self.modified_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea)

        try:
            moments = cv2.moments(contours[-2])
            self.x = int(moments['m10'] / moments['m00'])
            self.y = int(moments['m01'] / moments['m00'])
        except IndexError:
            pass

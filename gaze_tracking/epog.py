#!/usr/bin/env python3


"""
Demonstration of the eye point of gaze (EPOG) tracking library.

Call like this:
>> ./epog.py 1 'log_file_prefix'

'1': stabilize estimated EPOG w.r.t. previous cluster of EPOGs
'0': allow spurious EPOGs that deviate from cluster (default)

'log_file_prefix': (e.g. user_id) A logfile will be created with the errors, i.e.
the Euclidean distance (in pixels) between test points and corresponding estimated EPOGs.
Log file will be e.g. test_errors/'log_file_prefix'_stab_01-12-2019_18.36.44.txt
If log_file_prefix is omitted, log file will not be created.

Check the README.md for complete documentation.
"""

import gaze_tracking as gt
import cv2
import datetime
import os
import logging


class EPOG(object):

    def __init__(self, argv):
        self.logger = logging.getLogger(__name__)

        self.stabilize = False
        if len(argv) > 1:
            if argv[1] == '1':
                self.stabilize = True

        self.test_error_file = self.setup_test_error_file(argv)

        # iris_calib_window = self.setup_iris_calib_window()
        self.gaze_calib_window = self.setup_gaze_calib_window()
        self.windows_closed = False
        self.monitor = gt.get_screensize()
        self.webcam = None

        self.iris_calib = gt.IrisCalibration()
        self.gaze_tr = gt.GazeTracking(self.iris_calib)
        self.gaze_calib = gt.GazeCalibration(self.gaze_tr, self.monitor, self.test_error_file)
        self.pog = gt.PointOfGaze(self.gaze_tr, self.gaze_calib, self.monitor, self.stabilize)

    def setup_test_error_file(self, argv):
        test_error_file = None
        if len(argv) > 2:
            prefix = argv[2]
            test_error_dir = '../../GazeEvaluation/test_errors/'
            if not os.path.isdir(test_error_dir):
                os.makedirs(test_error_dir)
            if self.stabilize is True:
                test_error_file = open(test_error_dir + prefix + '_stab_' +
                                       datetime.datetime.now().strftime("%d-%m-%Y_%H.%M.%S") + '.txt', 'w+')
                self.logger.info('Stabilize: {}'.format(self.stabilize))
                self.logger.info('Log test errors in file: {}'.format(test_error_file))
            elif self.stabilize is False:
                test_error_file = open(test_error_dir + prefix + '_raw_' +
                                       datetime.datetime.now().strftime("%d-%m-%Y_%H.%M.%S") + '.txt', 'w+')
            self.logger.info('Stabilize: {}'.format(self.stabilize))
            self.logger.info('Logging test errors in: {}'.format(test_error_file.name))
            return test_error_file
        else:
            self.logger.info('Stabilize: {}'.format(self.stabilize))
            return None

    @staticmethod
    def setup_calib_window():
        """
        Window setup (window will be adjusted to the webcam-frame size,
        and will hold the annotated face of the user)
        """
        cv2.namedWindow('Iris', cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow('Iris', 0, 0)
        return 'Iris'

    @staticmethod
    def setup_gaze_calib_window():
        """
        Window setup (window is full-screen and will hold the calibration frame,
        displaying a series of red calibration points.)
        """
        cv2.namedWindow('Calibration', cv2.WINDOW_AUTOSIZE)
        cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        return 'Calibration'

    def setup_webcam(self):
        self.webcam = cv2.VideoCapture(0)
        webcam_w = int(self.webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
        webcam_h = int(self.webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.logger.info('Webcam resolution: {} x {}'.format(webcam_w, webcam_h))
        return self.webcam

    def analyze(self, frame):
        # We send this frame to GazeTracking to analyze for gaze direction
        self.gaze_tr.refresh(frame)

        screen_x, screen_y = None, None

        # calibrate iris_detection and annotate frame with pupil-landmarks
        if not self.iris_calib.is_complete():
            # rect = cv2.getWindowImageRect(iris_window)
            # cv2.moveWindow(iris_window, -rect[0], -rect[1])
            cam_frame = self.gaze_tr.annotated_frame()
            # cv2.imshow(iris_window, cam_frame)

        # calibrate the mapping from pupil to screen coordinates
        elif not self.gaze_calib.is_completed():
            # cv2.destroyWindow(iris_window)
            rect = cv2.getWindowImageRect(self.gaze_calib_window)
            cv2.moveWindow(self.gaze_calib_window, -rect[0], -rect[1])
            calib_frame = self.gaze_calib.calibrate_gaze()
            cv2.imshow(self.gaze_calib_window, calib_frame)

        # test the mapping
        elif not self.gaze_calib.is_tested():
            calib_frame = self.gaze_calib.test_gaze(self.pog)
            cv2.imshow(self.gaze_calib_window, calib_frame)

        # continue to unobtrusively estimate eye point of gaze
        else:
            if not self.windows_closed:
                # get the calibration window out of the way
                cv2.destroyAllWindows()
                self.windows_closed = True
            screen_x, screen_y = self.pog.point_of_gaze()

        return screen_x, screen_y

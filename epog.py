#!/usr/bin/env python


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

from __future__ import division
import sys
import cv2
from gaze_tracking import GazeTracking
from gaze_tracking.gazecalibration import GazeCalibration
from gaze_tracking.iriscalibration import IrisCalibration
from screeninfo import get_monitors
import datetime
import os

stabilize = False
if sys.argv.__len__() > 1:
    if sys.argv[1] == '1':
        stabilize = True

test_error_file = None
if sys.argv.__len__() > 2:
    prefix = sys.argv[2]
    if not os.path.isdir('test_errors'):
        os.makedirs('test_errors')
    if stabilize is True:
        test_error_file = open('test_errors/' + prefix + '_stab_' +
                               datetime.datetime.now().strftime("%d-%m-%Y_%H.%M.%S") + '.txt', 'w+')
        print('Stabilize: ', stabilize, 'Log test errors in file: ', test_error_file)
    elif stabilize is False:
        test_error_file = open('test_errors/' + prefix + '_raw_' +
                               datetime.datetime.now().strftime("%d-%m-%Y_%H.%M.%S") + '.txt', 'w+')
    print('Stabilize: ', stabilize, 'Log test errors in file: ', test_error_file)
else:
    print('Stabilize: ', stabilize)


def setup_iris_calib_window():
    """Window setup (window will be adjusted to the webcam-frame size,
    and will hold the annotated face of the user)
    """
    cv2.namedWindow('Iris', cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow('Iris', 0, 0)
    return 'Iris'


def setup_gaze_calib_window():
    """
    Window setup (window is full-screen and will hold the calibration frame,
    displaying a series of red calibration points.)
    """
    cv2.namedWindow('Calibration', cv2.WINDOW_AUTOSIZE)
    cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    return 'Calibration'


# Monitor ex: (x=0, y=0, width=1440, height=900, name=None)
monitor = get_monitors()[0]
calib_window = setup_gaze_calib_window()
webcam = cv2.VideoCapture(0)
# iris_window = setup_iris_calib_window()
windows_closed = False
iris_calib = IrisCalibration()
gaze = GazeTracking(iris_calib, monitor, stabilize)
gaze_calib = GazeCalibration(webcam, monitor, test_error_file)

while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()
    if frame is not None:

        # We send this frame to GazeTracking to analyze it
        gaze.refresh(frame)

        # calibrate iris_detection and annotate frame with pupil-landmarks
        if not iris_calib.is_complete():
            # rect = cv2.getWindowImageRect(iris_window)
            # cv2.moveWindow(iris_window, -rect[0], -rect[1])
            cam_frame = gaze.annotated_frame()
            # cv2.imshow(iris_window, cam_frame)
        # calibrate the mapping from pupil to screen coordinates
        elif not gaze_calib.is_completed():
            # cv2.destroyWindow(iris_window)
            rect = cv2.getWindowImageRect(calib_window)
            cv2.moveWindow(calib_window, -rect[0], -rect[1])
            calib_frame = gaze_calib.calibrate_gaze(gaze)
            cv2.imshow(calib_window, calib_frame)
        # test the mapping
        elif not gaze_calib.is_tested():
            calib_frame = gaze_calib.test_gaze(gaze)
            cv2.imshow(calib_window, calib_frame)
        # get the calibration window out of the way
        elif not windows_closed:
            cv2.destroyAllWindows()
            windows_closed = True
            break  # TODO: remove this line when in production
        # continue to unobtrusively estimate eye point of gaze
        else:
            screen_x, screen_y = gaze.point_of_gaze(gaze_calib)
            if screen_x is not None and screen_y is not None:
                pass  # or instead do something useful with the EPOG data

        if cv2.waitKey(1) == 27:  # Press Esc to quit
            # Release video capture
            webcam.release()
            cv2.destroyAllWindows()
            break

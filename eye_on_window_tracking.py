#!/usr/local/bin/python3.7

# Demonstration of the GazeTracking library.
# Check the README.md for complete documentation.

from __future__ import division
import cv2
from gaze_tracking import GazeTracking
from gaze_tracking.gazecalibration import GazeCalibration


gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
gaze_calibration = GazeCalibration(webcam)

while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    # annotate frame with pupil-landmarks
    frame = gaze.annotated_frame()
    if not gaze_calibration.is_calib_completed():
        gaze_calibration.calibrate_gaze(frame, gaze)
    elif not gaze_calibration.is_test_completed():
        gaze_calibration.test_gaze(frame, gaze)
    else:
        gaze.point_of_gaze(gaze_calibration)

    cv2.imshow('Calibration', frame)

    if cv2.waitKey(1) == 27:
        cv2.namedWindow('Calibration')
        break

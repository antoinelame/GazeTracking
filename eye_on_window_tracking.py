#!/usr/local/bin/python3.7

# Demonstration of the GazeTracking library.
# Check the README.md for complete documentation.

from __future__ import division
import cv2
from gaze_tracking import GazeTracking
from gaze_tracking.gazecalibration import GazeCalibration


gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
gaze_calib = GazeCalibration(webcam)
cv2.namedWindow('Calibration')

while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()
    if frame is not None:

        # We send this frame to GazeTracking to analyze it
        gaze.refresh(frame)

        # annotate frame with pupil-landmarks
        frame = gaze.annotated_frame()
        if not gaze_calib.is_calib_completed():
            gaze_calib.calibrate_gaze(frame, gaze)
        elif not gaze_calib.is_test_completed():
            gaze_calib.test_gaze(frame, gaze)
        else:
            gaze.point_of_gaze(gaze_calib)

        cv2.imshow('Calibration', frame)

        if cv2.waitKey(1) == 27:  # Esc
            # When everything done, release video capture
            webcam.release()
            cv2.destroyAllWindows()
            break

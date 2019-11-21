#!/Users/ritko75/Documents/forskning/Smartwork/WP4/ML/sw_data_coll/eye_tr_3.7/bin/python3.7

# Demonstration of the GazeTracking library.
# Check the README.md for complete documentation.

from __future__ import division
import cv2
from gaze_tracking import GazeTracking
from gaze_tracking.gazecalibration import GazeCalibration
from gaze_tracking.iriscalibration import IrisCalibration
from screeninfo import get_monitors


# Display-window setup (window is full-screen and will hold the annotated calibration frame)
def setup_iris_calib_window():
    cv2.namedWindow('Iris', cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow('Iris', 0, 0)
    return 'Iris'


# Display-window setup (window is full-screen and will hold the annotated calibration frame)
def setup_gaze_calib_window(monitor):
    cv2.namedWindow('Calibration', cv2.WINDOW_AUTOSIZE)
    cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    return 'Calibration'


# Monitor ex: (x=0, y=0, width=1440, height=900, name=None)
monitor = get_monitors()[0]
calib_window = setup_gaze_calib_window(monitor)
webcam = cv2.VideoCapture(0)
iris_window = setup_iris_calib_window()
windows_closed = False
iris_calib = IrisCalibration()
gaze = GazeTracking(iris_calib)
gaze_calib = GazeCalibration(webcam, monitor)

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
            cv2.imshow(iris_window, cam_frame)
        # calibrate the mapping from pupil to screen coordinates
        elif not gaze_calib.is_completed():
            cv2.destroyWindow(iris_window)
            rect = cv2.getWindowImageRect(calib_window)
            cv2.moveWindow(calib_window, -rect[0], -rect[1])
            calib_frame = gaze_calib.calibrate_gaze(gaze)
            cv2.imshow(calib_window, calib_frame)
        # test the mapping
        elif not gaze_calib.is_tested():
            calib_frame = gaze_calib.test_gaze(gaze)
            cv2.imshow(calib_window, calib_frame)
        # track eye point of gaze on the screen
        elif not windows_closed:
            cv2.destroyAllWindows()
            windows_closed = True
            break
        # continue to unobtrusively estimate eye point of gaze
        else:
            gaze.point_of_gaze(gaze_calib)

        if cv2.waitKey(1) == 27:  # Esc
            # When everything done, release video capture
            webcam.release()
            cv2.destroyAllWindows()
            break

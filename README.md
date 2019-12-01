# Gaze Tracking

![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)
![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
[![GitHub stars](https://img.shields.io/github/stars/antoinelame/GazeTracking.svg?style=social)](https://github.com/antoinelame/GazeTracking/stargazers)

This is a Python (2 and 3) library that provides a **webcam-based eye tracking system**. It gives you the exact position of the pupils and the gaze direction, in real time.

[![Demo](https://i.imgur.com/WNqgQkO.gif)](https://youtu.be/YEZMk1P0-yw)

## Installation

Clone this project:

```
git clone https://github.com/antoinelame/GazeTracking.git
```

In case you want to version handle this project in your own repo, you will need to use git-lfs to track the large .dat-file 
that is the trained face recognition model used for detecting facial landmarks. 
Install git-lfs: https://gitlab.ida.liu.se/help/workflow/lfs/manage_large_binaries_with_git_lfs.md

Install these dependencies (NumPy, OpenCV, Dlib), as well as other dependencies:

```
pip install -r requirements.txt
```

> The Dlib library has four primary prerequisites: Boost, Boost.Python, CMake and X11/XQuartx. If you do not have them, you can [read this article](https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/) to know how to easily install them.

Run the demo:

```
python epog.py
```

## Simple Demo

```python
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
```

## Documentation

In the following examples, `gaze` refers to an instance of the `GazeTracking` class.

### Refresh the frame

```python
gaze.refresh(frame)
```

Pass the frame to analyze (numpy.ndarray). If you want to work with a video stream, you need to put this instruction in a loop, like the example above.

### Position of the left pupil

```python
gaze.pupil_left_coords()
```

Returns the coordinates (x,y) of the left pupil.

### Position of the right pupil

```python
gaze.pupil_right_coords()
```

Returns the coordinates (x,y) of the right pupil.

### Looking to the left

```python
gaze.is_left()
```

Returns `True` if the user is looking to the left.

### Looking to the right

```python
gaze.is_right()
```

Returns `True` if the user is looking to the right.

### Looking at the center

```python
gaze.is_center()
```

Returns `True` if the user is looking at the center.

### Horizontal direction of the gaze

```python
ratio = gaze.horizontal_ratio()
```

Returns a number between 0.0 and 1.0 that indicates the horizontal direction of the gaze. The extreme right is 0.0, the center is 0.5 and the extreme left is 1.0.

### Vertical direction of the gaze

```python
ratio = gaze.vertical_ratio()
```

Returns a number between 0.0 and 1.0 that indicates the vertical direction of the gaze. The extreme top is 0.0, the center is 0.5 and the extreme bottom is 1.0.

### Blinking

```python
gaze.is_blinking()
```

Returns `True` if the user's eyes are closed.

### Webcam frame

```python
frame = gaze.annotated_frame()
```

Returns the main frame with pupils highlighted.

## You want to help?

Your suggestions, bugs reports and pull requests are welcome and appreciated. You can also starring ⭐️ the project!

If the detection of your pupils is not completely optimal, you can send me a video sample of you looking in different directions. I would use it to improve the algorithm.

## Licensing

This project is released by Antoine Lamé under the terms of the MIT Open Source License. View LICENSE for more information.

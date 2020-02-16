# Gaze Tracking

![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)
![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
[![GitHub stars](https://img.shields.io/github/stars/antoinelame/GazeTracking.svg?style=social)](https://github.com/antoinelame/GazeTracking/stargazers)

This is a Python (2 and 3) library that provides a **webcam-based eye tracking system**. It gives you the exact position of the pupils and the gaze direction, in real time.

[![Demo](https://i.imgur.com/WNqgQkO.gif)](https://youtu.be/YEZMk1P0-yw)

In addition, you can map pupil position onto screen coordinates, for example, to determine which window the user is looking at.

[![EPOG_demo](https://i.imgur.com/8LxBNQE.gif)](https://i.imgur.com/8LxBNQE.gif)

User is fixating at the red dots. The small white dots mark the EPOG estimate.
## Installation

Clone this project:

```
git clone https://github.com/antoinelame/GazeTracking.git
```

In case you want to version handle this project in your own repo, you will need to use git-lfs to track the large .dat-file 
that is the trained face recognition model used for detecting facial landmarks. 
Install git-lfs: https://gitlab.ida.liu.se/help/workflow/lfs/manage_large_binaries_with_git_lfs.md

Install dependencies (NumPy, OpenCV, Dlib), as well as other dependencies:

```
pip install -r requirements.txt
```

> The Dlib library has four primary prerequisites: Boost, Boost.Python, CMake and X11/XQuartx. If you do not have them, you can [read this article](https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/) to know how to easily install them.

In addition, if you want screen-size handling:
```
pip install pypiwin32  # for Windows
```
```
pip install pyobjc  # for MacOS
```
Screen-size handling in MacOS also requires AppKit, which is included in XCode.
```
pip install python3-xlib  # for Linux
```

Run the demo:

```
./epog_example.py
```

## Simple Demo

```python
#!/usr/bin/env python3


"""
Demonstration of how to use the eye point of gaze (EPOG) tracking library.

This example application can be called like this (both args are optional):
>> ./epog_example.py 1 'log_file_prefix'

'1': stabilize estimated EPOG w.r.t. previous cluster of EPOGs
'0': allow spurious EPOGs that deviate from cluster (default)

'log_file_prefix': (e.g. user_id) A logfile will be created with the errors, i.e.
the Euclidean distance (in pixels) between test points and corresponding estimated EPOGs.
Log file will be e.g. test_errors/'log_file_prefix'_stab_01-12-2019_18.36.44.txt
If log_file_prefix is omitted, log file will not be created.

Check the README.md for complete documentation.
"""

import sys
import cv2
import gaze_tracking as gt

# setup_epog expects max two args, both optional,
# sets up webcam, and calibration windows
test_error_dir = '../GazeEvaluation/test_errors/'
epog = gt.EPOG(test_error_dir, sys.argv)


while True:
    # We get a new frame from the webcam
    _, frame = epog.webcam.read()
    if frame is not None:
        # Analyze gaze direction and map to screen coordinates
        screen_x, screen_y = epog.analyze(frame)

        # Access gaze direction
        text = ""
        if epog.gaze_tr.is_right():
            text = "Looking right"
        elif epog.gaze_tr.is_left():
            text = "Looking left"
        elif epog.gaze_tr.is_center():
            text = "Looking center"

        # Use gaze projected onto screen surface
        # Screen coords will be None for a few initial frames,
        # before calibration and tests have been completed
        if screen_x is not None and screen_y is not None:
            text = "Looking at point {}, {} on the screen".format(screen_x, screen_y)

        # Press Esc to quit the video analysis loop
        if cv2.waitKey(1) == 27:
            # Release video capture
            epog.webcam.release()
            cv2.destroyAllWindows()
            break
        # Note: The waitkey function is the only method in HighGUI that can fetch and handle events,
        # so it needs to be called periodically for normal event processing unless HighGUI
        # is used within an environment that takes care of event processing.
        # Note: The waitkey function only works if there is at least one HighGUI window created and
        # the window is active. If there are several HighGUI windows, any of them can be active.
        # (https://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html)

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

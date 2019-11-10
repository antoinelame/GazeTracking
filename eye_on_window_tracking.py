#!/usr/local/bin/python3.7

# Demonstration of the GazeTracking library.
# Check the README.md for complete documentation.


import cv2
from gaze_tracking import GazeTracking

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
# coordinates for positioning of output text in the demo frame
left_x = 90
center_x = 450
right_x = 750
up_y = 60
level_y = 300
down_y = 500
(x, y) = (center_x, level_y)
# for displaying calibration points
screen_size = (1280, 720)
circle_rad = 20
calib_points = [(circle_rad, circle_rad),
                (screen_size[0] - circle_rad, circle_rad),
                (screen_size[0] - circle_rad, screen_size[1] - circle_rad),
                (circle_rad, screen_size[1] - circle_rad),
                (screen_size[0]//2 - circle_rad, screen_size[1]//2 - circle_rad)]
screen_corner = 0
t = 1
cv2.namedWindow('Calibration', cv2.WINDOW_KEEPRATIO)
cv2.setWindowProperty('Calibration', cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    text = ""

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (x, y+40), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (x, y+75), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    if gaze.is_blinking():
        text = "blinking"
    else:
        if gaze.is_right():
            text = "right "
            x = right_x
        elif gaze.is_left():
            text = "left "
            x = left_x
        elif gaze.is_center():
            text = "center "
            x = center_x

    if not gaze.is_blinking():
        if gaze.is_up():
            text = text + "up"
            y = up_y
        elif gaze.is_down():
            text = text + "down"
            y = down_y
        elif gaze.is_level():
            text = text + "level"
            y = level_y

    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    if screen_corner < len(calib_points):
        # draw a red calibration circle. Params: center, rad, color, ..
        cv2.circle(frame, calib_points[screen_corner], circle_rad, (0, 0, 255), -1)
        if t % 25 == 0:
            screen_corner = screen_corner + 1
            t = 1
        else:
            t = t + 1

    cv2.imshow('Calibration', frame)

    if cv2.waitKey(1) == 27:
        break

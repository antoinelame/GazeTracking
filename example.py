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
right_x = 850
up_y = 60
center_y = 200
down_y = 400

while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    text = ""

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    if gaze.is_blinking():
        text = "Blinking"
        cv2.putText(frame, text, (450, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
    elif gaze.is_right():
        text = "Looking right"
        cv2.putText(frame, text, (850, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
    elif gaze.is_left():
        text = "Looking left"
        cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
    elif gaze.is_center():
        text = "Looking center"
        cv2.putText(frame, text, (450, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        break

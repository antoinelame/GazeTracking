"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""
import datetime
import os
import cv2
from gaze_tracking import GazeTracking
import time
import utils
import numpy as np

stim_dir = "left"
stim_dist = "23"
stim_trial = "XX"
stim_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
savefile = f"{stim_dir}_{stim_dist}_{stim_trial}_{stim_time}.npy"

print_interval = 1000 # n frames to update to console
new_width = 400
new_height = 300

center = (new_width, new_height)
x = center[1] / 2 - new_width / 2
y = center[0] / 2 - new_height / 2

gaze = GazeTracking()
webcam = cv2.VideoCapture(max(utils.list_webcams()))
init_time = time.perf_counter()
framecount = 0

# TODO: get monitor resolution, adjust params
monitor_res_y = 1080
monitor_res_x = 720
patient_distance = 40 # cm

study_eye = "left"
study_eye_positions = []

while True:
    loop_start_time = time.perf_counter()
    framecount += 1

    # We get a new frame from the webcam
    _, frame = webcam.read()

    # Resize the image
    frame = cv2.resize(frame, (new_width, new_height))
    crop_img = frame[int(y):int(y + new_height), int(x):int(x + new_width)]

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame) # this is the most expensive operation
    frame = gaze.annotated_frame()
    text = ""

    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    if left_pupil is None and right_pupil is None:
        print ("No pupils detected")
    if study_eye == "left":
        if left_pupil is not None:
            study_eye_positions.append(left_pupil)
            deviation = utils.distance_from_fixation((crop_img.shape[0] // 2, crop_img.shape[1] // 2), left_pupil)
            # study_eye_deviations.append(deviation)
    elif study_eye == "right":
        if right_pupil is not None:
            study_eye_positions.append(right_pupil)
            deviation = utils.distance_from_fixation((crop_img.shape[0] // 2, crop_img.shape[1] // 2), right_pupil)
            # study_eye_deviations.append(deviation)
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        break
    fps = 1/(time.perf_counter() - loop_start_time)
    if framecount % print_interval == 0:
        print(f"FPS: {fps}")

study_eye_deviations = [utils.distance_from_fixation((crop_img.shape[0] // 2, crop_img.shape[1] // 2), pos) for pos in study_eye_positions]
print ("Average deviation of study eye from fixation: ", np.round(np.mean(study_eye_deviations), 4), "over ", framecount, "frames")
print ("Std dev of deviation of study eye from fixation: ", np.round(np.std(study_eye_deviations), 4), "over ", framecount, "frames")
print (os.getcwd())
np.save(os.path.join("data", savefile), study_eye_positions)
webcam.release()
cv2.destroyAllWindows()

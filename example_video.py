"""
Demonstration of the GazeTracking library for video inference.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking
import argparse
import os

# Parser
def parser():
    parser = argparse.ArgumentParser(description="GazeTracking Video Inference.")
    parser.add_argument("--vid", type=str, required=True, help="path to video")
    parser.add_argument("--output", type=str, default="result.avi", help="inference video name. Only support .avi extension due to OpenCV 3.4")
    parser.add_argument("--dont_show", action='store_true', help="hide imshow window")
    return parser.parse_args()

def check_arguments_errors(args):
    if not (os.path.isfile(args.vid)):
        raise(ValueError("Invalid video path {}".format(os.path.abspath(args.video))))

# Save Video
def save_video(input_video, output_video, size):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video

# Main
def capture_main():
    cap = cv2.VideoCapture(args.vid)

    # Getting width and heights
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #Video Saver
    video = save_video(cap, args.output, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame is not None:

            #Process Inference
            image = inference(frame)

            if not args.dont_show:
                cv2.imshow('Inference', image)
            
            video.write(image)
            
            if cv2.waitKey(1) == 27:
                break    
    cap.release()
    video.release()
    cv2.destroyAllWindows()

# Drawing
def inference(frame):
    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

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

    frame = cv2.putText(frame, text, (int(frame.shape[1]*0.01), int(frame.shape[0]*0.05)), cv2.FONT_HERSHEY_DUPLEX, 1, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    frame = cv2.putText(frame, "Left pupil:  " + str(left_pupil), (int(frame.shape[1]*0.01), int(frame.shape[0]*0.1)), cv2.FONT_HERSHEY_DUPLEX, 0.8, (147, 58, 31), 1)
    frame = cv2.putText(frame, "Right pupil: " + str(right_pupil), (int(frame.shape[1]*0.01), int(frame.shape[0]*0.15)), cv2.FONT_HERSHEY_DUPLEX, 0.8, (147, 58, 31), 1)
    return frame


if __name__ == '__main__':
    args = parser()
    check_arguments_errors(args)

    gaze = GazeTracking()

    print('='*30)
    print('Input Video Loaded: '+str(args.vid))
    capture_main()
    print('Inference Video Saved to: '+str(args.output))
    print('='*30)
import sieve
import cv2
from gaze_tracking import GazeTracking
import os
import tempfile
import time

@sieve.function(
    name="gaze-tracker",
    python_version="3.9",
    python_packages=[
        "opencv-python",
        "mediapipe",
        "numpy"
    ],
    system_packages=["ffmpeg"],
)
def gaze_tracker(video: sieve.File) -> list:
    """
    Accepts a Sieve video file or YouTube link, returns metadata with intervals for center gaze.
    Output: list of dicts: {start_time: float, end_time: float, is_center: bool}
    """
    # Download video if it's a YouTube link
    video_path = video.path
    if video.url and ("youtube.com" in video.url or "youtu.be" in video.url):
        youtube_downloader = sieve.function.get("sieve/youtube-downloader")
        video = next(youtube_downloader.push(video.url).result())
        video_path = video.path

    cap = cv2.VideoCapture(video_path)
    gaze = GazeTracking()
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0

    intervals = []
    current_state = None  # None, 'center', 'not_center'
    start_time = 0
    last_time = 0
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gaze.refresh(frame)
        is_center = gaze.is_center() is True
        timestamp = frame_idx / fps if fps > 0 else 0

        if current_state is None:
            current_state = 'center' if is_center else 'not_center'
            start_time = timestamp
        elif (is_center and current_state == 'not_center') or (not is_center and current_state == 'center'):
            intervals.append({
                'start_time': start_time,
                'end_time': timestamp,
                'is_center': current_state == 'center'
            })
            current_state = 'center' if is_center else 'not_center'
            start_time = timestamp
        last_time = timestamp
        frame_idx += 1

    if current_state is not None and last_time > start_time:
        intervals.append({
            'start_time': start_time,
            'end_time': last_time,
            'is_center': current_state == 'center'
        })
    cap.release()
    return intervals 
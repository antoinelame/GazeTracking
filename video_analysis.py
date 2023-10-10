import os
import cv2
import matplotlib.pyplot as plt
from gaze_tracking import GazeTracking
from config import *
from tqdm import tqdm
import logging
import datetime
import argparse
import json

log = logging.getLogger("video_analysis")
logging.basicConfig(level = logging.INFO)

def calculate_circle_position(value, image_width):
    # Define the left and right positions for the circle
    left_position = MIN_RATIO
    right_position = MAX_RATIO

    # Ensure the value is within the range [0.4, 0.6]
    value = max(left_position, min(right_position, value))

    # Calculate the horizontal position of the circle using linear interpolation
    circle_x = int((value - left_position) / (right_position - left_position) * image_width)

    return circle_x


def analyze_video(path, max_x: int, output_path_video: str, show=False):
    gaze = GazeTracking()
    cap = cv2.VideoCapture(path)
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    points = []
    if output_path_video:
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        size = (frame_width, frame_height)
        result = cv2.VideoWriter(output_path_video,
                                 cv2.VideoWriter_fourcc(*'MJPG'),
                                 10, size)
    with tqdm(total=total_frame_count - 3) as pbar:
        while cap.isOpened(): # if too slow on better computer, perhaps need to apply multithread
            ret, frame = cap.read()
            if ret:
                gaze.refresh(frame)
                ratio = gaze.horizontal_ratio()
                if ratio:
                    circle_x = calculate_circle_position(ratio, max_x)
                    points.append(circle_x)
                    if show:
                        frame = cv2.circle(frame, (circle_x, int(frame.shape[0] / 2)), radius=20, color=(0, 0, 255),
                                           thickness=-1)
                if output_path_video:
                    result.write(frame)
                if show:
                    cv2.imshow("Demo", frame)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                pbar.update(1)
            else:
                break
        cap.release()
    if output_path_video:
        result.release()
    if show:
        cv2.destroyAllWindows()
    return points


def show_histogram(point_list: list, max_x: int) -> None:
    n, bins, patches = plt.hist(point_list, density=False, bins=5)  # density=False would make counts
    plt.bar_label(patches)
    plt.ylabel('Number of occurrence')
    plt.xlim(xmax=max_x, xmin=0)
    plt.xlabel('')
    plt.show()


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    elif value.lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Invalid boolean value: {}".format(value))


def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Script for analyzing where a person is looking at the screen")

    # Add the -p or --path argument
    parser.add_argument('-p', '--path', required=False, help="path to video clip", default='data/sample_input.mp4')
    parser.add_argument('-o', '--output', required=False, default='data/output.json',
                        help="path to output json")
    parser.add_argument('-s', '--show', required=False, type=str_to_bool, default=False,
                        help='Show the video during processing with dot drawn')
    parser.add_argument('-x', '--max_x', required=False, default=1920,
                        help='Max horizontal resolution of the TV,default 1920')
    parser.add_argument('--hist', required=False, type=str_to_bool, default=False,
                        help='Whether to show histogram after processing')
    parser.add_argument('--video_output', required=False, default=None,
                        help='Path where video output (with dot printed should be saved')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the file path argument
    file_path = args.path
    output_path = args.output
    show = args.show
    max_x = args.max_x
    show_hist = args.hist
    output_path_video = args.video_output

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f'No file under location: {file_path}.Please check the path provided')

    if not output_path.lower().endswith(".json"):
        raise TypeError('output file must be of json format')
    output_directory = os.path.dirname(output_path)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    if output_path_video:
        output_video = os.path.dirname(output_path_video)
        if output_video:
            os.makedirs(output_video, exist_ok=True)
    log.info('Starting Analysis...')
    spotted_points = analyze_video(path=file_path, max_x=max_x, output_path_video=output_path_video, show=show)
    log.info('Analysis completed!')
    if show_hist:
        show_histogram(spotted_points, max_x=max_x)

    output = {
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'input_file': file_path,
        'points': spotted_points
    }

    json_str = json.dumps(output, indent=4)

    # Write the JSON string to the output file
    with open(output_path, "w") as output_file:
        output_file.write(json_str)

    log.info(f'points spotted saved under {output_path} directory.')
    return spotted_points


if __name__ == "__main__":
    points = main()

import cv2
from gaze_tracking import GazeTracking
import os


def calculate_circle_position(value, image_width):
    # Define the left and right positions for the circle
    left_position = 0.4
    right_position = 0.6

    # Ensure the value is within the range [0.4, 0.6]
    value = max(left_position, min(right_position, value))

    # Calculate the horizontal position of the circle using linear interpolation
    circle_x = int((value - left_position) / (right_position - left_position) * image_width)

    return circle_x
def test_all_images(images, GLASSES = True):
    gaze = GazeTracking()
    for image in images:
        path = f'data/{image}.jpg'
        frame = cv2.imread(path)
        scale_percent = 50
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        gaze.refresh(frame)
        ratio = gaze.horizontal_ratio()
        if ratio:
            # print(f'for path: {path} ratio: {ratio}')
            frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
            circle_x = calculate_circle_position(ratio, frame.shape[1])
            # print(f"The circle should be at x={circle_x} in the image.")
            frame = cv2.circle(frame, (circle_x,int(frame.shape[0]/2)), radius=20, color=(0, 0, 255), thickness=-1)
            cv2.imshow("Demo", frame)

        k = cv2.waitKey(0) & 0xFF
        if k == 27:  # wait for ESC key to exit
            cv2.destroyAllWindows()
        if GLASSES:
            path = f'data/glass_{image}.jpg'
            frame = cv2.imread(path)
            gaze.refresh(frame)
            ratio = gaze.horizontal_ratio()
            print(f'for path: {path} ratio: {ratio}')
            circle_x = calculate_circle_position(ratio, frame.shape[1])
            print(f"The circle should be at x={circle_x} in the image.")

def test_video(path, save=True):
    gaze = GazeTracking()
    cap = cv2.VideoCapture(path)
    if save:
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        size = (frame_width, frame_height)
        result = cv2.VideoWriter('filename.avi',
                                 cv2.VideoWriter_fourcc(*'MJPG'),
                                 10, size)
    while (cap.isOpened()):
        _, frame = cap.read()
        gaze.refresh(frame)
        ratio = gaze.horizontal_ratio()
        # print(f'for path: {path} ratio: {ratio}')
        circle_x = calculate_circle_position(ratio, frame.shape[1])
        # print(f"The circle should be at x={circle_x} in the image.")
        frame = cv2.circle(frame, (circle_x, int(frame.shape[0] / 2)), radius=20, color=(0, 0, 255), thickness=-1)
        if save:
            result.write(frame)
        # cv2.imshow("Demo", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    if save:
        result.release()
    # Closes all the frames
    cv2.destroyAllWindows()

test_all_images(['right','center','left'])
# test_video('data/movie.mp4')


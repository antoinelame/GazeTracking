

import cv2

def list_webcams():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr


def distance_from_fixation(origin, point):
    return ((origin[0] - point[0]) ** 2 + (origin[1] - point[1]) ** 2) ** 0.5

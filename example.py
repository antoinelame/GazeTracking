"""
Simple gaze direction test - shows ratio values for calibration
"""

import cv2
from gaze_tracking import GazeTracking

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

print("Gaze Direction Test")
print("Look left, center, and right to see the ratio values")
print("Press 'q' to quit")

while True:
    _, frame = webcam.read()
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    
    # Get and display ratio
    ratio = gaze.horizontal_ratio()
    if ratio is not None:
        ratio_text = f"Ratio: {ratio:.3f}"
        cv2.putText(frame, ratio_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show pupil positions for debugging
        if gaze.eye_left and gaze.eye_right and gaze.eye_left.pupil and gaze.eye_right.pupil:
            left_abs = gaze.eye_left.origin[0] + gaze.eye_left.pupil.x
            right_abs = gaze.eye_right.origin[0] + gaze.eye_right.pupil.x
            cv2.putText(frame, f"Left: {left_abs:.0f}, Right: {right_abs:.0f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Determine direction
        if ratio <= 0.4:
            direction = "RIGHT"
            color = (0, 0, 255)  # Red
        elif ratio >= 0.6:
            direction = "LEFT"
            color = (255, 0, 0)  # Blue
        else:
            direction = "CENTER"
            color = (0, 255, 0)  # Green
            
        cv2.putText(frame, direction, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    else:
        cv2.putText(frame, "No eyes detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Gaze Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
   
webcam.release()
cv2.destroyAllWindows() 
"""
Gaze calibration script - shows ratio values for fine-tuning
"""

import cv2
from gaze_tracking import GazeTracking

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

print("Gaze Calibration")
print("Look left, center, and right with just your eyes (keep head still)")
print("Watch the ratio values change")
print("Press 'q' to quit")

while True:
    _, frame = webcam.read()
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    
    # Show ratio and direction
    ratio = gaze.horizontal_ratio()
    if ratio is not None:
        # Show ratio value
        ratio_text = f"Ratio: {ratio:.3f}"
        cv2.putText(frame, ratio_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show current direction
        if gaze.is_right():
            direction = "RIGHT"
            color = (0, 0, 255)  # Red
        elif gaze.is_left():
            direction = "LEFT" 
            color = (255, 0, 0)  # Blue
        elif gaze.is_center():
            direction = "CENTER"
            color = (0, 255, 0)  # Green
        else:
            direction = "UNKNOWN"
            color = (255, 255, 0)  # Yellow
            
        cv2.putText(frame, direction, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        # Show thresholds
        cv2.putText(frame, "Thresholds: <=0.45=RIGHT, 0.45-0.55=CENTER, >=0.55=LEFT", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show instructions
        cv2.putText(frame, "Look left/right with eyes only (keep head still)", 
                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    else:
        cv2.putText(frame, "No eyes detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Gaze Calibration", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
   
webcam.release()
cv2.destroyAllWindows() 
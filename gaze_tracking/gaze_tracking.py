"""
Enhanced GazeTracking class using modular tracker backends.
Supports multiple tracking methods (DNN, Haar, Hybrid) via unified API.
"""

from __future__ import division
import cv2
import math
import numpy as np
from .eye import Eye
from .calibration import Calibration
from .trackers import create_tracker, BaseTracker


class GazeTracking(object):
    """
    This class tracks the user's gaze using modular tracker backends.
    It provides useful information like the position of the eyes
    and pupils and allows to know if the eyes are open or closed.
    Enhanced with pupil diameter, gaze angle, and face detection.
    """

    def __init__(self, tracker_type='dnn', tracker=None):
        """
        Initialize GazeTracking.
        
        Args:
            tracker_type: Type of tracker ('dnn', 'haar', 'hybrid'). Default: 'dnn'
            tracker: Optional pre-initialized tracker instance
        """
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = Calibration()
        self.face_bbox = None
        self.face_detected = False
        
        # Initialize tracker
        if tracker is not None:
            if not isinstance(tracker, BaseTracker):
                raise ValueError("Tracker must be an instance of BaseTracker")
            self.tracker = tracker
        else:
            self.tracker = create_tracker(tracker_type)
        
        self.tracker_type = tracker_type

    @property
    def pupils_located(self):
        """Check that the pupils have been located"""
        try:
            if self.eye_left is None or self.eye_right is None:
                return False
            if self.eye_left.pupil is None or self.eye_right.pupil is None:
                return False
            if self.eye_left.pupil.x is None or self.eye_left.pupil.y is None:
                return False
            if self.eye_right.pupil.x is None or self.eye_right.pupil.y is None:
                return False
            return True
        except Exception:
            return False

    def _analyze(self):
        """Detects the face and initialize Eye objects using the tracker"""
        if self.frame is None:
            self.eye_left = None
            self.eye_right = None
            self.face_detected = False
            return
        
        # Detect face
        self.face_bbox = self.tracker.detect_face(self.frame)
        self.face_detected = self.face_bbox is not None
        
        if not self.face_detected:
            self.eye_left = None
            self.eye_right = None
            return
        
        # Detect eyes
        eyes = self.tracker.detect_eyes(self.frame, self.face_bbox)
        
        # Initialize Eye objects
        try:
            # Left eye
            left_eye_region = eyes.get('left_eye')
            if left_eye_region is not None and left_eye_region.size > 0:
                left_coords = self.tracker.get_eye_region_coords(self.face_bbox, 'left')
                try:
                    self.eye_left = Eye(
                        self.frame,
                        eye_region=left_eye_region,
                        side=0,
                        calibration=self.calibration,
                        eye_coords=left_coords
                    )
                except Exception as e:
                    # If Eye initialization fails, try with just coordinates
                    self.eye_left = None
            else:
                # Fallback: use region-based detection
                left_coords = self.tracker.get_eye_region_coords(self.face_bbox, 'left')
                left_eye_region = self.tracker.extract_eye_region(self.frame, left_coords)
                if left_eye_region is not None and left_eye_region.size > 0:
                    try:
                        self.eye_left = Eye(
                            self.frame,
                            eye_region=left_eye_region,
                            side=0,
                            calibration=self.calibration,
                            eye_coords=left_coords
                        )
                    except:
                        self.eye_left = None
                else:
                    self.eye_left = None
            
            # Right eye
            right_eye_region = eyes.get('right_eye')
            if right_eye_region is not None and right_eye_region.size > 0:
                right_coords = self.tracker.get_eye_region_coords(self.face_bbox, 'right')
                try:
                    self.eye_right = Eye(
                        self.frame,
                        eye_region=right_eye_region,
                        side=1,
                        calibration=self.calibration,
                        eye_coords=right_coords
                    )
                except Exception as e:
                    self.eye_right = None
            else:
                # Fallback: use region-based detection
                right_coords = self.tracker.get_eye_region_coords(self.face_bbox, 'right')
                right_eye_region = self.tracker.extract_eye_region(self.frame, right_coords)
                if right_eye_region is not None and right_eye_region.size > 0:
                    try:
                        self.eye_right = Eye(
                            self.frame,
                            eye_region=right_eye_region,
                            side=1,
                            calibration=self.calibration,
                            eye_coords=right_coords
                        )
                    except:
                        self.eye_right = None
                else:
                    self.eye_right = None
        except Exception as e:
            # Silent fail - don't print errors in production
            self.eye_left = None
            self.eye_right = None

    def refresh(self, frame):
        """Refreshes the frame and analyzes it.

        Arguments:
            frame (numpy.ndarray): The frame to analyze
        """
        self.frame = frame
        self._analyze()

    def is_face_detected(self):
        """Returns True if a face is detected in the current frame"""
        return self.face_detected

    def pupil_left_coords(self):
        """Returns the coordinates of the left pupil"""
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return (x, y)
        return None

    def pupil_right_coords(self):
        """Returns the coordinates of the right pupil"""
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return (x, y)
        return None

    def pupil_left_diameter(self):
        """Returns the diameter of the left pupil in pixels"""
        if self.pupils_located and self.eye_left.pupil.diameter is not None:
            return self.eye_left.pupil.diameter
        return None

    def pupil_right_diameter(self):
        """Returns the diameter of the right pupil in pixels"""
        if self.pupils_located and self.eye_right.pupil.diameter is not None:
            return self.eye_right.pupil.diameter
        return None

    def pupil_diameter(self):
        """Returns the average diameter of both pupils in pixels"""
        left_d = self.pupil_left_diameter()
        right_d = self.pupil_right_diameter()
        
        if left_d is not None and right_d is not None:
            return (left_d + right_d) / 2.0
        elif left_d is not None:
            return left_d
        elif right_d is not None:
            return right_d
        return None

    def horizontal_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        horizontal direction of the gaze. The extreme right is 0.0,
        the center is 0.5 and the extreme left is 1.0
        """
        if self.pupils_located:
            try:
                pupil_left = self.eye_left.pupil.x / (self.eye_left.center[0] * 2 - 10)
                pupil_right = self.eye_right.pupil.x / (self.eye_right.center[0] * 2 - 10)
                return (pupil_left + pupil_right) / 2
            except (ZeroDivisionError, TypeError):
                return None
        return None

    def vertical_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        vertical direction of the gaze. The extreme top is 0.0,
        the center is 0.5 and the extreme bottom is 1.0
        """
        if self.pupils_located:
            try:
                pupil_left = self.eye_left.pupil.y / (self.eye_left.center[1] * 2 - 10)
                pupil_right = self.eye_right.pupil.y / (self.eye_right.center[1] * 2 - 10)
                return (pupil_left + pupil_right) / 2
            except (ZeroDivisionError, TypeError):
                return None
        return None

    def gaze_angle(self):
        """
        Calculate gaze angle in degrees.
        
        Returns:
            Tuple of (horizontal_angle, vertical_angle) in degrees, or None
        """
        h_ratio = self.horizontal_ratio()
        v_ratio = self.vertical_ratio()
        
        if h_ratio is None or v_ratio is None:
            return None
        
        # Convert ratios to angles (approximate, can be calibrated)
        # Assuming ±30 degrees horizontal and ±20 degrees vertical range
        horizontal_angle = (h_ratio - 0.5) * 60.0  # -30 to +30 degrees
        vertical_angle = (v_ratio - 0.5) * 40.0  # -20 to +20 degrees
        
        return (horizontal_angle, vertical_angle)

    def is_right(self):
        """Returns true if the user is looking to the right"""
        if self.pupils_located:
            ratio = self.horizontal_ratio()
            if ratio is not None:
                return ratio <= 0.35
        return False

    def is_left(self):
        """Returns true if the user is looking to the left"""
        if self.pupils_located:
            ratio = self.horizontal_ratio()
            if ratio is not None:
                return ratio >= 0.65
        return False

    def is_center(self):
        """Returns true if the user is looking to the center"""
        if self.pupils_located:
            return self.is_right() is not True and self.is_left() is not True
        return False

    def is_blinking(self):
        """Returns true if the user closes his eyes (binary classification)"""
        if not self.pupils_located:
            return True  # If pupils not located, assume eyes are closed
        
        try:
            # Use EAR (Eye Aspect Ratio) if available
            if self.eye_left.ear is not None and self.eye_right.ear is not None:
                avg_ear = (self.eye_left.ear + self.eye_right.ear) / 2.0
                # EAR threshold: <0.2 typically indicates closed
                return avg_ear < 0.2
            else:
                # Fallback to blinking ratio
                if self.eye_left.blinking is not None and self.eye_right.blinking is not None:
                    blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2
                    return blinking_ratio > 3.8
        except (AttributeError, TypeError):
            pass
        
        return False

    def eye_state(self):
        """
        Get binary eye state classification.
        
        Returns:
            1 for open, 0 for closed
        """
        return 0 if self.is_blinking() else 1

    def eye_left_center(self):
        """Returns the center coordinates of the left eye region"""
        if self.eye_left is not None and self.eye_left.origin is not None:
            x = self.eye_left.origin[0] + self.eye_left.center[0]
            y = self.eye_left.origin[1] + self.eye_left.center[1]
            return (int(x), int(y))
        return None

    def eye_right_center(self):
        """Returns the center coordinates of the right eye region"""
        if self.eye_right is not None and self.eye_right.origin is not None:
            x = self.eye_right.origin[0] + self.eye_right.center[0]
            y = self.eye_right.origin[1] + self.eye_right.center[1]
            return (int(x), int(y))
        return None

    def annotated_frame(self):
        """Returns the main frame with pupils highlighted and annotations"""
        frame = self.frame.copy() if self.frame is not None else None
        
        if frame is None:
            return None

        # Draw face bounding box
        if self.face_detected and self.face_bbox is not None:
            x, y, w, h = self.face_bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Draw eye regions
        if self.eye_left is not None and self.eye_left.origin is not None:
            eye_x, eye_y = self.eye_left.origin
            eye_w = self.eye_left.frame.shape[1] if self.eye_left.frame is not None else 50
            eye_h = self.eye_left.frame.shape[0] if self.eye_left.frame is not None else 30
            cv2.rectangle(frame, (eye_x, eye_y), (eye_x + eye_w, eye_y + eye_h), (255, 255, 0), 1)
            
            # Draw eye center
            center = self.eye_left_center()
            if center:
                cv2.circle(frame, center, 3, (255, 255, 0), -1)
        
        if self.eye_right is not None and self.eye_right.origin is not None:
            eye_x, eye_y = self.eye_right.origin
            eye_w = self.eye_right.frame.shape[1] if self.eye_right.frame is not None else 50
            eye_h = self.eye_right.frame.shape[0] if self.eye_right.frame is not None else 30
            cv2.rectangle(frame, (eye_x, eye_y), (eye_x + eye_w, eye_y + eye_h), (255, 255, 0), 1)
            
            # Draw eye center
            center = self.eye_right_center()
            if center:
                cv2.circle(frame, center, 3, (255, 255, 0), -1)

        # Draw pupils in green
        if self.pupils_located:
            color = (0, 255, 0)  # Green in BGR
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()
            
            if x_left is not None and y_left is not None:
                # Draw crosshair
                cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color, 2)
                cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color, 2)
                # Draw circle for pupil
                if self.eye_left.pupil.diameter is not None and self.eye_left.pupil.diameter > 0:
                    radius = max(3, int(self.eye_left.pupil.diameter / 2))
                    cv2.circle(frame, (x_left, y_left), radius, color, 2)
                else:
                    # Default circle if diameter not available
                    cv2.circle(frame, (x_left, y_left), 5, color, 2)
            
            if x_right is not None and y_right is not None:
                # Draw crosshair
                cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color, 2)
                cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color, 2)
                # Draw circle for pupil
                if self.eye_right.pupil.diameter is not None and self.eye_right.pupil.diameter > 0:
                    radius = max(3, int(self.eye_right.pupil.diameter / 2))
                    cv2.circle(frame, (x_right, y_right), radius, color, 2)
                else:
                    # Default circle if diameter not available
                    cv2.circle(frame, (x_right, y_right), 5, color, 2)

        return frame

    def switch_tracker(self, tracker_type):
        """
        Switch to a different tracker method.
        
        Args:
            tracker_type: Type of tracker ('dnn', 'haar', 'hybrid')
        """
        self.tracker = create_tracker(tracker_type)
        self.tracker_type = tracker_type

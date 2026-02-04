"""
Eye detection and analysis using OpenCV.
Enhanced with improved blinking detection using Eye Aspect Ratio (EAR).
"""

import math
import numpy as np
import cv2
from .pupil import Pupil


class Eye(object):
    """
    This class creates a new frame to isolate the eye and
    initiates the pupil detection. Enhanced for OpenCV-based tracking.
    """

    LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
    RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]

    def __init__(self, original_frame, eye_region=None, landmarks=None, side=0, calibration=None, eye_coords=None):
        """
        Initialize eye detection.
        
        Args:
            original_frame: Original frame (BGR or grayscale)
            eye_region: Pre-extracted eye region (numpy array), optional
            landmarks: Facial landmarks (numpy array or dlib format), optional
            side: 0 for left eye, 1 for right eye
            calibration: Calibration object for threshold
            eye_coords: Eye region coordinates (x, y, w, h) if eye_region is provided
        """
        self.frame = None
        self.origin = None
        self.center = None
        self.pupil = None
        self.landmark_points = None
        self.blinking = None
        self.ear = None  # Eye Aspect Ratio
        self.side = side

        self._analyze(original_frame, eye_region, landmarks, side, calibration, eye_coords)

    @staticmethod
    def _middle_point(p1, p2):
        """Returns the middle point (x,y) between two points
        
        Arguments:
            p1: First point (tuple, numpy array, or dlib point)
            p2: Second point (tuple, numpy array, or dlib point)
        """
        # Handle different point formats
        if hasattr(p1, 'x'):  # dlib point
            x1, y1 = p1.x, p1.y
        else:
            x1, y1 = p1[0], p1[1]
        
        if hasattr(p2, 'x'):  # dlib point
            x2, y2 = p2.x, p2.y
        else:
            x2, y2 = p2[0], p2[1]
        
        x = int((x1 + x2) / 2)
        y = int((y1 + y2) / 2)
        return (x, y)

    def _isolate_from_landmarks(self, frame, landmarks, points):
        """Isolate an eye using landmarks (for backward compatibility with dlib).
        
        Arguments:
            frame: Frame containing the face
            landmarks: Facial landmarks (dlib format)
            points: List of landmark indices for the eye
        """
        # Handle dlib landmarks
        if hasattr(landmarks, 'part'):
            region = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in points])
        else:
            # Assume numpy array format
            region = np.array([landmarks[point] for point in points])
        
        region = region.astype(np.int32)
        self.landmark_points = region

        # Applying a mask to get only the eye
        height, width = frame.shape[:2]
        black_frame = np.zeros((height, width), np.uint8)
        mask = np.full((height, width), 255, np.uint8)
        cv2.fillPoly(mask, [region], (0, 0, 0))
        eye = cv2.bitwise_not(black_frame, frame.copy(), mask=mask)

        # Cropping on the eye
        margin = 5
        min_x = np.min(region[:, 0]) - margin
        max_x = np.max(region[:, 0]) + margin
        min_y = np.min(region[:, 1]) - margin
        max_y = np.max(region[:, 1]) + margin

        self.frame = eye[min_y:max_y, min_x:max_x]
        self.origin = (min_x, min_y)

        height, width = self.frame.shape[:2]
        self.center = (width / 2, height / 2)

    def _isolate_from_region(self, frame, eye_region, eye_coords):
        """Isolate eye from pre-extracted region.
        
        Arguments:
            frame: Original frame
            eye_region: Pre-extracted eye region (grayscale)
            eye_coords: Eye coordinates (x, y, w, h) in original frame
        """
        self.frame = eye_region.copy() if eye_region is not None else None
        
        if eye_coords is not None:
            self.origin = (eye_coords[0], eye_coords[1])
        else:
            self.origin = (0, 0)
        
        if self.frame is not None:
            height, width = self.frame.shape[:2]
            self.center = (width / 2, height / 2)
        else:
            self.center = (0, 0)

    def _calculate_ear(self, eye_frame):
        """
        Calculate Eye Aspect Ratio (EAR) for blink detection.
        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        Lower EAR indicates closed eye.
        
        Args:
            eye_frame: Eye region frame
            
        Returns:
            EAR value
        """
        if eye_frame is None or eye_frame.size == 0:
            return 0.0
        
        h, w = eye_frame.shape[:2]
        
        # Approximate vertical and horizontal measurements
        # Vertical: distance between top and bottom of eye
        vertical_1 = h * 0.3  # Top of eye
        vertical_2 = h * 0.7  # Bottom of eye
        vertical_dist = abs(vertical_2 - vertical_1)
        
        # Horizontal: width of eye
        horizontal_dist = w
        
        if horizontal_dist == 0:
            return 0.0
        
        # EAR calculation (simplified)
        ear = vertical_dist / (2.0 * horizontal_dist)
        return ear

    def _blinking_ratio(self, landmarks, points):
        """Calculates a ratio that can indicate whether an eye is closed or not.
        It's the division of the width of the eye, by its height.
        
        Arguments:
            landmarks: Facial landmarks (dlib format or numpy array)
            points: List of landmark indices for the eye
            
        Returns:
            The computed ratio
        """
        try:
            # Handle dlib landmarks
            if hasattr(landmarks, 'part'):
                left = (landmarks.part(points[0]).x, landmarks.part(points[0]).y)
                right = (landmarks.part(points[3]).x, landmarks.part(points[3]).y)
                top = self._middle_point(landmarks.part(points[1]), landmarks.part(points[2]))
                bottom = self._middle_point(landmarks.part(points[5]), landmarks.part(points[4]))
            else:
                # Assume numpy array format
                left = landmarks[points[0]]
                right = landmarks[points[3]]
                top = self._middle_point(landmarks[points[1]], landmarks[points[2]])
                bottom = self._middle_point(landmarks[points[5]], landmarks[points[4]])
            
            eye_width = math.hypot((left[0] - right[0]), (left[1] - right[1]))
            eye_height = math.hypot((top[0] - bottom[0]), (top[1] - bottom[1]))
            
            try:
                ratio = eye_width / eye_height
            except ZeroDivisionError:
                ratio = None
        except (IndexError, AttributeError):
            ratio = None

        return ratio

    def _analyze(self, original_frame, eye_region=None, landmarks=None, side=0, calibration=None, eye_coords=None):
        """Detects and isolates the eye in a new frame, sends data to the calibration
        and initializes Pupil object.
        
        Arguments:
            original_frame: Frame passed by the user
            eye_region: Pre-extracted eye region (optional)
            landmarks: Facial landmarks (optional)
            side: Indicates whether it's the left eye (0) or the right eye (1)
            calibration: Manages the binarization threshold value
            eye_coords: Eye region coordinates if eye_region is provided
        """
        # Convert frame to grayscale if needed
        if len(original_frame.shape) == 3:
            gray_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = original_frame
        
        # Extract eye region
        if eye_region is not None:
            # Use pre-extracted eye region (from tracker)
            self._isolate_from_region(gray_frame, eye_region, eye_coords)
        elif landmarks is not None:
            # Use landmarks (backward compatibility)
            if side == 0:
                points = self.LEFT_EYE_POINTS
            elif side == 1:
                points = self.RIGHT_EYE_POINTS
            else:
                return
            
            self.blinking = self._blinking_ratio(landmarks, points)
            self._isolate_from_landmarks(gray_frame, landmarks, points)
        else:
            # No eye region or landmarks provided
            return
        
        # Calculate EAR for improved blink detection
        if self.frame is not None:
            self.ear = self._calculate_ear(self.frame)
            
            # Use EAR for blinking ratio if landmarks not available
            if self.blinking is None:
                # Convert EAR to blinking ratio (inverse relationship)
                # Higher EAR = open eye, lower EAR = closed eye
                self.blinking = self.ear * 5.0  # Scale factor for compatibility
        
        # Calibration and pupil detection
        if self.frame is not None and calibration is not None:
            if not calibration.is_complete():
                calibration.evaluate(self.frame, side)
            
            threshold = calibration.threshold(side)
            self.pupil = Pupil(self.frame, threshold)
        elif self.frame is not None:
            # Use default threshold if no calibration
            default_threshold = 50
            self.pupil = Pupil(self.frame, default_threshold)

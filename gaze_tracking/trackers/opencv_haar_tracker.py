"""
OpenCV Haar Cascade-based tracker for face and eye detection.
This is a non-ML method using traditional computer vision.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
from .base_tracker import BaseTracker


class OpenCVHaarTracker(BaseTracker):
    """
    OpenCV Haar Cascade-based tracker.
    Fast, non-ML method using traditional computer vision techniques.
    """
    
    def __init__(self):
        """Initialize the Haar Cascade tracker."""
        super().__init__()
        self.name = "OpenCV_Haar"
        
        # Load Haar Cascade classifiers
        self.face_cascade = None
        self.eye_cascade = None
        self._load_cascades()
        
        self.is_initialized = self.face_cascade is not None
    
    def _load_cascades(self):
        """Load Haar Cascade classifiers from OpenCV's data directory."""
        import os
        
        # Helper function to find cascade file
        def find_cascade(filename):
            paths = []
            
            # First, try local trained_models directory (most reliable)
            base_dir = os.path.abspath(os.path.dirname(__file__))
            local_path = os.path.join(base_dir, '..', 'trained_models', 'haarcascades', filename)
            paths.append(os.path.abspath(local_path))
            
            # Try cv2.data.haarcascades if available
            try:
                if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
                    paths.append(cv2.data.haarcascades + filename)
            except:
                pass
            
            # Try common installation paths
            possible_paths = [
                '/usr/local/share/opencv4/haarcascades/' + filename,
                '/usr/share/opencv/haarcascades/' + filename,
                os.path.join(os.path.dirname(cv2.__file__), 'data', filename),
            ]
            paths.extend(possible_paths)
            
            # Try each path
            for path in paths:
                if os.path.exists(path):
                    cascade = cv2.CascadeClassifier(path)
                    if not cascade.empty():
                        return cascade
            
            # Try loading without path (OpenCV might find it)
            cascade = cv2.CascadeClassifier(filename)
            if not cascade.empty():
                return cascade
            
            return None
        
        try:
            # Face detection cascade
            self.face_cascade = find_cascade('haarcascade_frontalface_default.xml')
            
            # Eye detection cascade (try with glasses support first)
            self.eye_cascade = find_cascade('haarcascade_eye_tree_eyeglasses.xml')
            
            # Fallback to regular eye cascade if glasses version fails
            if self.eye_cascade is None or self.eye_cascade.empty():
                self.eye_cascade = find_cascade('haarcascade_eye.xml')
            
            if self.face_cascade is None or (hasattr(self.face_cascade, 'empty') and self.face_cascade.empty()):
                print("Warning: Could not load face cascade")
            if self.eye_cascade is None or (hasattr(self.eye_cascade, 'empty') and self.eye_cascade.empty()):
                print("Warning: Could not load eye cascade")
        except Exception as e:
            print(f"Error loading Haar Cascades: {e}")
    
    def detect_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect face using Haar Cascade.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Face bounding box (x, y, width, height) or None
        """
        if self.face_cascade is None or self.face_cascade.empty():
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # Detect faces with more sensitive parameters
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,  # Reduced from 5 for better detection
            minSize=(50, 50),  # Increased minimum size for better accuracy
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) > 0:
            # Return the largest face
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            return tuple(largest_face)
        
        return None
    
    def detect_eyes(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Dict[str, Optional[np.ndarray]]:
        """
        Detect eye regions within the face bounding box using Haar Cascade.
        
        Args:
            frame: Input frame (BGR format)
            face_bbox: Face bounding box (x, y, width, height)
            
        Returns:
            Dictionary with 'left_eye' and 'right_eye' regions
        """
        if self.eye_cascade is None or self.eye_cascade.empty():
            # Fallback to region-based approach
            return self._detect_eyes_region_based(frame, face_bbox)
        
        x, y, w, h = face_bbox
        
        # Extract face region
        face_region = frame[y:y+h, x:x+w]
        
        if face_region.size == 0:
            return {'left_eye': None, 'right_eye': None}
        
        # Convert to grayscale
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY) if len(face_region.shape) == 3 else face_region
        
        # Detect eyes in face region with more sensitive parameters
        eyes = self.eye_cascade.detectMultiScale(
            gray_face,
            scaleFactor=1.1,
            minNeighbors=2,  # Reduced for better detection
            minSize=(15, 15)  # Smaller minimum size to catch more eyes
        )
        
        left_eye = None
        right_eye = None
        
        if len(eyes) >= 2:
            # Sort eyes by x-coordinate (left eye has smaller x)
            eyes_sorted = sorted(eyes, key=lambda e: e[0])
            
            # Left eye (first in sorted list)
            ex, ey, ew, eh = eyes_sorted[0]
            left_eye_region = gray_face[ey:ey+eh, ex:ex+ew]
            if left_eye_region.size > 0:
                left_eye = left_eye_region
            
            # Right eye (second in sorted list)
            if len(eyes_sorted) > 1:
                ex, ey, ew, eh = eyes_sorted[1]
                right_eye_region = gray_face[ey:ey+eh, ex:ex+ew]
                if right_eye_region.size > 0:
                    right_eye = right_eye_region
        elif len(eyes) == 1:
            # Only one eye detected, use region-based for the other
            return self._detect_eyes_region_based(frame, face_bbox)
        else:
            # No eyes detected, use region-based approach
            return self._detect_eyes_region_based(frame, face_bbox)
        
        return {'left_eye': left_eye, 'right_eye': right_eye}
    
    def _detect_eyes_region_based(self, frame: np.ndarray, 
                                  face_bbox: Tuple[int, int, int, int]) -> Dict[str, Optional[np.ndarray]]:
        """
        Region-based eye detection when cascade fails.
        Uses approximate eye positions based on face geometry.
        """
        left_coords = self.get_eye_region_coords(face_bbox, 'left')
        right_coords = self.get_eye_region_coords(face_bbox, 'right')
        
        left_eye = self.extract_eye_region(frame, left_coords)
        right_eye = self.extract_eye_region(frame, right_coords)
        
        return {'left_eye': left_eye, 'right_eye': right_eye}
    
    def detect_pupils(self, eye_frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Detect pupil coordinates using OpenCV contour analysis and HoughCircles.
        
        Args:
            eye_frame: Grayscale eye region
            
        Returns:
            Pupil coordinates (x, y) relative to eye region, or None
        """
        if eye_frame is None or eye_frame.size == 0:
            return None
        
        # Method 1: Try HoughCircles first (good for circular pupils)
        circles = cv2.HoughCircles(
            eye_frame,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=eye_frame.shape[0] // 2,
            param1=50,
            param2=30,
            minRadius=eye_frame.shape[0] // 8,
            maxRadius=eye_frame.shape[0] // 3
        )
        
        if circles is not None and len(circles) > 0:
            # Use the first (largest) circle
            circle = circles[0][0]
            cx, cy = int(circle[0]), int(circle[1])
            return (cx, cy)
        
        # Method 2: Fallback to contour analysis
        # Apply bilateral filter
        filtered = cv2.bilateralFilter(eye_frame, 10, 15, 15)
        
        # Apply adaptive thresholding
        threshold = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
        threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find the largest contour (likely the pupil)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate centroid
        M = cv2.moments(largest_contour)
        
        if M["m00"] == 0:
            return None
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        return (cx, cy)
    
    def get_eye_state(self, eye_frame: np.ndarray) -> int:
        """
        Classify eye state as Open (1) or Closed (0) using EAR and contour analysis.
        
        Args:
            eye_frame: Grayscale eye region
            
        Returns:
            1 for open, 0 for closed
        """
        if eye_frame is None or eye_frame.size == 0:
            return 0
        
        # Calculate Eye Aspect Ratio
        ear = self.calculate_eye_aspect_ratio(eye_frame)
        
        # Additional check: contour area ratio
        # Closed eyes have less visible contour area
        threshold_img = cv2.adaptiveThreshold(
            eye_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        contours, _ = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            total_area = sum(cv2.contourArea(c) for c in contours)
            frame_area = eye_frame.shape[0] * eye_frame.shape[1]
            area_ratio = total_area / frame_area if frame_area > 0 else 0
        else:
            area_ratio = 0
        
        # Combined threshold: EAR and area ratio
        ear_threshold = 0.2
        area_threshold = 0.1
        
        is_open = (ear > ear_threshold) and (area_ratio > area_threshold)
        
        return 1 if is_open else 0
    
    def get_landmarks(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Get approximate facial landmarks based on face geometry.
        Haar Cascade doesn't provide landmarks, so we estimate them.
        
        Args:
            frame: Input frame
            face_bbox: Face bounding box
            
        Returns:
            Array of estimated landmark points
        """
        x, y, w, h = face_bbox
        
        # Estimate landmarks based on face geometry
        landmarks = np.array([
            [x + w * 0.2, y + h * 0.4],   # Left eye center
            [x + w * 0.8, y + h * 0.4],   # Right eye center
            [x + w * 0.5, y + h * 0.6],   # Nose tip
            [x + w * 0.3, y + h * 0.8],   # Left mouth corner
            [x + w * 0.7, y + h * 0.8],   # Right mouth corner
        ])
        
        return landmarks

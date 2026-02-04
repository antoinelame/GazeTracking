"""
Base tracker interface for modular eye tracking backends.
All tracker implementations must inherit from this abstract base class.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Dict
import numpy as np
import cv2


class BaseTracker(ABC):
    """
    Abstract base class defining the interface for all eye tracking backends.
    This ensures a unified API regardless of the underlying detection method.
    """
    
    def __init__(self):
        """Initialize the tracker."""
        self.name = self.__class__.__name__
        self.is_initialized = False
    
    @abstractmethod
    def detect_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect face in the frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (x, y, width, height) bounding box, or None if no face detected
        """
        pass
    
    @abstractmethod
    def detect_eyes(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Dict[str, Optional[np.ndarray]]:
        """
        Detect eye regions within the face bounding box.
        
        Args:
            frame: Input frame (BGR format)
            face_bbox: Face bounding box (x, y, width, height)
            
        Returns:
            Dictionary with keys 'left_eye' and 'right_eye', each containing
            the eye region as numpy array, or None if not detected
        """
        pass
    
    @abstractmethod
    def detect_pupils(self, eye_frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Detect pupil coordinates in an eye region.
        
        Args:
            eye_frame: Grayscale eye region frame
            
        Returns:
            Tuple of (x, y) pupil coordinates relative to eye region, or None if not detected
        """
        pass
    
    @abstractmethod
    def get_eye_state(self, eye_frame: np.ndarray) -> int:
        """
        Classify eye state as Open (1) or Closed (0).
        
        Args:
            eye_frame: Grayscale eye region frame
            
        Returns:
            1 for open, 0 for closed
        """
        pass
    
    def get_landmarks(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Get facial landmarks if available.
        
        Args:
            frame: Input frame (BGR format)
            face_bbox: Face bounding box (x, y, width, height)
            
        Returns:
            Array of landmark points, or None if not available
        """
        return None
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for detection (optional override).
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Preprocessed frame
        """
        return frame
    
    def get_eye_region_coords(self, face_bbox: Tuple[int, int, int, int], 
                              eye_side: str) -> Tuple[int, int, int, int]:
        """
        Calculate approximate eye region coordinates within face bounding box.
        This is a utility method that can be overridden by specific trackers.
        
        Args:
            face_bbox: Face bounding box (x, y, width, height)
            eye_side: 'left' or 'right'
            
        Returns:
            Eye region bounding box (x, y, width, height) relative to full frame
        """
        x, y, w, h = face_bbox
        
        # Improved eye region estimation based on face geometry
        # Eyes are typically in the upper 1/3 to 1/2 of the face
        eye_height = max(20, int(h * 0.2))  # Minimum 20 pixels
        eye_y = y + int(h * 0.25)  # Start at 25% from top
        eye_width = max(30, int(w * 0.25))  # Minimum 30 pixels, 25% of face width
        
        if eye_side == 'left':
            eye_x = x + int(w * 0.12)  # Left eye: 12% from left edge
        else:  # right
            eye_x = x + int(w * 0.63)  # Right eye: 63% from left edge (leaving space for nose)
        
        return (eye_x, eye_y, eye_width, eye_height)
    
    def extract_eye_region(self, frame: np.ndarray, 
                          coords: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract eye region from frame.
        
        Args:
            frame: Input frame
            coords: Eye region coordinates (x, y, width, height)
            
        Returns:
            Extracted eye region as grayscale
        """
        x, y, w, h = coords
        
        # Ensure coordinates are within frame bounds
        h_frame, w_frame = frame.shape[:2]
        x = max(0, min(x, w_frame - 1))
        y = max(0, min(y, h_frame - 1))
        w = max(1, min(w, w_frame - x))
        h = max(1, min(h, h_frame - y))
        
        # Extract region
        eye_region = frame[y:y+h, x:x+w]
        
        # Ensure we have a valid region
        if eye_region.size == 0:
            return None
        
        # Convert to grayscale if needed
        if len(eye_region.shape) == 3:
            eye_region = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        elif len(eye_region.shape) == 2:
            # Already grayscale
            pass
        else:
            return None
        
        return eye_region
    
    def calculate_eye_aspect_ratio(self, eye_frame: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) for blink detection.
        Higher EAR indicates open eye, lower indicates closed.
        
        Args:
            eye_frame: Grayscale eye region
            
        Returns:
            EAR value (typically 0.2-0.4 for open, <0.2 for closed)
        """
        h, w = eye_frame.shape[:2]
        
        # Approximate vertical and horizontal measurements
        # This is a simplified version - can be improved with landmarks
        vertical_1 = h * 0.3
        vertical_2 = h * 0.7
        horizontal = w
        
        if horizontal == 0:
            return 0.0
        
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear
    
    def is_face_in_frame(self, frame: np.ndarray) -> bool:
        """
        Check if a face is detected in the frame.
        
        Args:
            frame: Input frame
            
        Returns:
            True if face detected, False otherwise
        """
        face_bbox = self.detect_face(frame)
        return face_bbox is not None

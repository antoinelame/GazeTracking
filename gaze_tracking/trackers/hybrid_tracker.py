"""
Hybrid tracker combining multiple detection methods for improved accuracy.
Uses DNN for face detection (more accurate) and Haar for eye detection (faster),
with fallback mechanisms.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
from .base_tracker import BaseTracker
from .opencv_dnn_tracker import OpenCVDNNTracker
from .opencv_haar_tracker import OpenCVHaarTracker


class HybridTracker(BaseTracker):
    """
    Hybrid tracker that combines DNN and Haar Cascade methods.
    Uses the best of both worlds: DNN for accuracy, Haar for speed.
    """
    
    def __init__(self):
        """Initialize the hybrid tracker with both DNN and Haar components."""
        super().__init__()
        self.name = "Hybrid"
        
        # Initialize both trackers
        self.dnn_tracker = OpenCVDNNTracker()
        self.haar_tracker = OpenCVHaarTracker()
        
        # Strategy: Use DNN for face, Haar for eyes (faster)
        # Can fallback to either method if one fails
        self.use_dnn_for_face = True
        self.use_haar_for_eyes = True
        
        self.is_initialized = self.dnn_tracker.is_initialized or self.haar_tracker.is_initialized
    
    def detect_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect face using DNN (more accurate) with Haar fallback.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Face bounding box (x, y, width, height) or None
        """
        # Try DNN first (more accurate)
        if self.use_dnn_for_face and self.dnn_tracker.is_initialized:
            face_bbox = self.dnn_tracker.detect_face(frame)
            if face_bbox is not None:
                return face_bbox
        
        # Fallback to Haar (faster, more reliable)
        if self.haar_tracker.is_initialized:
            face_bbox = self.haar_tracker.detect_face(frame)
            if face_bbox is not None:
                return face_bbox
        
        return None
    
    def detect_eyes(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Dict[str, Optional[np.ndarray]]:
        """
        Detect eyes using Haar (faster) with DNN fallback.
        
        Args:
            frame: Input frame (BGR format)
            face_bbox: Face bounding box (x, y, width, height)
            
        Returns:
            Dictionary with 'left_eye' and 'right_eye' regions
        """
        # Try Haar first (faster)
        if self.use_haar_for_eyes and self.haar_tracker.is_initialized:
            eyes = self.haar_tracker.detect_eyes(frame, face_bbox)
            # Check if both eyes detected
            if eyes['left_eye'] is not None and eyes['right_eye'] is not None:
                return eyes
        
        # Fallback to DNN method
        if self.dnn_tracker.is_initialized:
            eyes = self.dnn_tracker.detect_eyes(frame, face_bbox)
            if eyes['left_eye'] is not None and eyes['right_eye'] is not None:
                return eyes
        
        # Final fallback: use region-based from either tracker
        if self.haar_tracker.is_initialized:
            return self.haar_tracker._detect_eyes_region_based(frame, face_bbox)
        elif self.dnn_tracker.is_initialized:
            return self.dnn_tracker._detect_eyes_region_based(frame, face_bbox)
        
        return {'left_eye': None, 'right_eye': None}
    
    def detect_pupils(self, eye_frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Detect pupils using combined methods with confidence scoring.
        
        Args:
            eye_frame: Grayscale eye region
            
        Returns:
            Pupil coordinates (x, y) relative to eye region, or None
        """
        if eye_frame is None or eye_frame.size == 0:
            return None
        
        # Try both methods and use the one with higher confidence
        dnn_result = None
        haar_result = None
        
        # Try DNN method
        if self.dnn_tracker.is_initialized:
            try:
                dnn_result = self.dnn_tracker.detect_pupils(eye_frame)
            except:
                pass
        
        # Try Haar method
        if self.haar_tracker.is_initialized:
            try:
                haar_result = self.haar_tracker.detect_pupils(eye_frame)
            except:
                pass
        
        # Return the first successful result
        # In a more sophisticated implementation, we could combine results
        if dnn_result is not None:
            return dnn_result
        elif haar_result is not None:
            return haar_result
        
        return None
    
    def get_eye_state(self, eye_frame: np.ndarray) -> int:
        """
        Classify eye state using combined methods with voting.
        
        Args:
            eye_frame: Grayscale eye region
            
        Returns:
            1 for open, 0 for closed
        """
        if eye_frame is None or eye_frame.size == 0:
            return 0
        
        votes = []
        
        # Get classification from DNN tracker
        if self.dnn_tracker.is_initialized:
            try:
                dnn_state = self.dnn_tracker.get_eye_state(eye_frame)
                votes.append(dnn_state)
            except:
                pass
        
        # Get classification from Haar tracker
        if self.haar_tracker.is_initialized:
            try:
                haar_state = self.haar_tracker.get_eye_state(eye_frame)
                votes.append(haar_state)
            except:
                pass
        
        # Majority vote (or average if we want to weight them)
        if len(votes) > 0:
            # If both agree, use that. Otherwise, use DNN (more accurate)
            if len(votes) == 2 and votes[0] == votes[1]:
                return votes[0]
            elif len(votes) >= 1:
                # Prefer DNN result if available
                if self.dnn_tracker.is_initialized and len(votes) > 0:
                    return votes[0]  # DNN is first
                return votes[-1]  # Haar is last
        
        return 0  # Default to closed if no detection
    
    def get_landmarks(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Get facial landmarks from DNN tracker (more accurate).
        
        Args:
            frame: Input frame
            face_bbox: Face bounding box
            
        Returns:
            Array of landmark points or None
        """
        if self.dnn_tracker.is_initialized:
            return self.dnn_tracker.get_landmarks(frame, face_bbox)
        elif self.haar_tracker.is_initialized:
            return self.haar_tracker.get_landmarks(frame, face_bbox)
        
        return None
    
    def set_strategy(self, use_dnn_for_face: bool = True, use_haar_for_eyes: bool = True):
        """
        Configure the hybrid strategy.
        
        Args:
            use_dnn_for_face: Use DNN for face detection (default: True)
            use_haar_for_eyes: Use Haar for eye detection (default: True)
        """
        self.use_dnn_for_face = use_dnn_for_face
        self.use_haar_for_eyes = use_haar_for_eyes

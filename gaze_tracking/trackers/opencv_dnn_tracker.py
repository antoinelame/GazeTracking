"""
OpenCV DNN-based tracker for face and eye detection.
This is the default ML-based method using deep neural networks.
"""

import os
import cv2
import numpy as np
from typing import Tuple, Optional, Dict
from .base_tracker import BaseTracker


class OpenCVDNNTracker(BaseTracker):
    """
    OpenCV DNN-based tracker using pre-trained deep learning models.
    Provides high accuracy face and eye detection.
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize the OpenCV DNN tracker.
        
        Args:
            model_dir: Directory containing model files. If None, uses default location.
        """
        super().__init__()
        self.name = "OpenCV_DNN"
        self.face_net = None
        self.confidence_threshold = 0.7
        
        # Try to load OpenCV's DNN face detector
        self._load_face_detector(model_dir)
        self.is_initialized = self.face_net is not None
    
    def _load_face_detector(self, model_dir: Optional[str] = None):
        """
        Load OpenCV DNN face detection model.
        Tries multiple common model locations and formats.
        """
        if model_dir is None:
            # Try to find models in trained_models directory
            base_dir = os.path.abspath(os.path.dirname(__file__))
            model_dir = os.path.join(base_dir, '..', 'trained_models', 'opencv_dnn')
        
        # Try OpenCV's built-in face detector (if available)
        # Otherwise, we'll use a fallback method
        try:
            # OpenCV 4.5.1+ has built-in DNN face detector
            # Try to load from OpenCV's samples
            prototxt_path = None
            model_path = None
            
            # Check for common model locations
            possible_paths = [
                (os.path.join(model_dir, 'deploy.prototxt'),
                 os.path.join(model_dir, 'res10_300x300_ssd_iter_140000.caffemodel')),
                (os.path.join(model_dir, 'opencv_face_detector.pbtxt'),
                 os.path.join(model_dir, 'opencv_face_detector_uint8.pb')),
            ]
            
            for prototxt, model in possible_paths:
                if os.path.exists(prototxt) and os.path.exists(model):
                    prototxt_path = prototxt
                    model_path = model
                    break
            
            if prototxt_path and model_path:
                self.face_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            elif prototxt_path and model_path and prototxt_path.endswith('.pbtxt'):
                self.face_net = cv2.dnn.readNetFromTensorflow(model_path, prototxt_path)
            else:
                # Use OpenCV's built-in DNN face detector (if available in newer versions)
                # Fallback: we'll use a simpler method
                self.face_net = None
        except Exception as e:
            print(f"Warning: Could not load DNN face detector: {e}")
            print("Falling back to Haar Cascade method for face detection")
            self.face_net = None
    
    def detect_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect face using OpenCV DNN.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Face bounding box (x, y, width, height) or None
        """
        if self.face_net is None:
            # Fallback to Haar Cascade if DNN not available
            return self._detect_face_haar_fallback(frame)
        
        h, w = frame.shape[:2]
        
        # Create blob from frame
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 
            1.0, 
            (300, 300), 
            [104, 117, 123]
        )
        
        # Set input and get detections
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        # Find the best detection
        best_confidence = 0
        best_box = None
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence_threshold and confidence > best_confidence:
                # Get bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                
                # Convert to (x, y, width, height) format
                x = max(0, x1)
                y = max(0, y1)
                width = min(w, x2) - x
                height = min(h, y2) - y
                
                if width > 0 and height > 0:
                    best_box = (x, y, width, height)
                    best_confidence = confidence
        
        return best_box
    
    def _detect_face_haar_fallback(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Fallback to Haar Cascade if DNN not available."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # Try to load Haar Cascade - handle different OpenCV installations
        cascade_paths = []
        
        # First, try local trained_models directory (most reliable)
        import os
        base_dir = os.path.abspath(os.path.dirname(__file__))
        local_path = os.path.join(base_dir, '..', 'trained_models', 'haarcascades', 'haarcascade_frontalface_default.xml')
        cascade_paths.append(os.path.abspath(local_path))
        
        # Try cv2.data.haarcascades if available
        try:
            # Check if cv2.data exists and has haarcascades attribute
            if hasattr(cv2, 'data'):
                data_module = getattr(cv2, 'data', None)
                if data_module and hasattr(data_module, 'haarcascades'):
                    haarcascades_path = getattr(data_module, 'haarcascades', None)
                    if haarcascades_path:
                        cascade_paths.append(haarcascades_path + 'haarcascade_frontalface_default.xml')
        except (AttributeError, TypeError):
            pass
        
        # Try common installation paths
        possible_paths = [
            '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
            os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascade_frontalface_default.xml'),
        ]
        cascade_paths.extend(possible_paths)
        
        face_cascade = None
        for path in cascade_paths:
            if os.path.exists(path):
                face_cascade = cv2.CascadeClassifier(path)
                if not face_cascade.empty():
                    break
        
        # If still no cascade found, try loading without path (OpenCV might find it)
        if face_cascade is None or face_cascade.empty():
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
        if face_cascade is None or face_cascade.empty():
            return None
        
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Return the largest face
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            return tuple(largest_face)
        
        return None
    
    def detect_eyes(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Dict[str, Optional[np.ndarray]]:
        """
        Detect eye regions within the face bounding box.
        Uses region-based approach with OpenCV processing.
        
        Args:
            frame: Input frame (BGR format)
            face_bbox: Face bounding box (x, y, width, height)
            
        Returns:
            Dictionary with 'left_eye' and 'right_eye' regions
        """
        x, y, w, h = face_bbox
        
        # Extract face region
        face_region = frame[y:y+h, x:x+w]
        
        if face_region.size == 0:
            return {'left_eye': None, 'right_eye': None}
        
        # Convert to grayscale
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY) if len(face_region.shape) == 3 else face_region
        
        # Use Haar Cascade for eye detection - find cascade file safely
        import os
        cascade_paths = []
        
        # First, try local trained_models directory
        base_dir = os.path.abspath(os.path.dirname(__file__))
        local_path = os.path.join(base_dir, '..', 'trained_models', 'haarcascades', 'haarcascade_eye.xml')
        cascade_paths.append(os.path.abspath(local_path))
        
        # Try cv2.data.haarcascades if available
        try:
            if hasattr(cv2, 'data'):
                data_module = getattr(cv2, 'data', None)
                if data_module and hasattr(data_module, 'haarcascades'):
                    haarcascades_path = getattr(data_module, 'haarcascades', None)
                    if haarcascades_path:
                        cascade_paths.append(haarcascades_path + 'haarcascade_eye.xml')
        except (AttributeError, TypeError):
            pass
        
        # Try common installation paths
        possible_paths = [
            '/usr/local/share/opencv4/haarcascades/haarcascade_eye.xml',
            '/usr/share/opencv/haarcascades/haarcascade_eye.xml',
            os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascade_eye.xml'),
        ]
        cascade_paths.extend(possible_paths)
        
        eye_cascade = None
        for path in cascade_paths:
            if os.path.exists(path):
                eye_cascade = cv2.CascadeClassifier(path)
                if not eye_cascade.empty():
                    break
        
        # If still no cascade, try loading without path
        if eye_cascade is None or eye_cascade.empty():
            eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        
        if eye_cascade is None or eye_cascade.empty():
            # Fallback to region-based approach
            return self._detect_eyes_region_based(frame, face_bbox)
        
        eyes = eye_cascade.detectMultiScale(gray_face, 1.1, 3)
        
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
        else:
            # Fallback to region-based approach
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
        Detect pupil coordinates using OpenCV contour analysis.
        
        Args:
            eye_frame: Grayscale eye region
            
        Returns:
            Pupil coordinates (x, y) relative to eye region, or None
        """
        if eye_frame is None or eye_frame.size == 0:
            return None
        
        # Apply bilateral filter to reduce noise
        filtered = cv2.bilateralFilter(eye_frame, 10, 15, 15)
        
        # Apply adaptive thresholding
        threshold = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morphological operations to clean up
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
        Classify eye state as Open (1) or Closed (0) using EAR.
        
        Args:
            eye_frame: Grayscale eye region
            
        Returns:
            1 for open, 0 for closed
        """
        if eye_frame is None or eye_frame.size == 0:
            return 0
        
        # Calculate Eye Aspect Ratio
        ear = self.calculate_eye_aspect_ratio(eye_frame)
        
        # Threshold for open/closed (can be calibrated)
        # Typical EAR: 0.2-0.4 for open, <0.2 for closed
        threshold = 0.2
        
        return 1 if ear > threshold else 0
    
    def get_landmarks(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Get facial landmarks if available.
        For DNN tracker, we can estimate landmarks from face geometry.
        
        Args:
            frame: Input frame
            face_bbox: Face bounding box
            
        Returns:
            Array of landmark points or None
        """
        # Simplified landmark estimation based on face geometry
        # In a full implementation, you would use a landmark detection model
        x, y, w, h = face_bbox
        
        landmarks = np.array([
            [x + w * 0.2, y + h * 0.4],   # Left eye center
            [x + w * 0.8, y + h * 0.4],   # Right eye center
            [x + w * 0.5, y + h * 0.6],   # Nose tip
            [x + w * 0.3, y + h * 0.8],   # Left mouth corner
            [x + w * 0.7, y + h * 0.8],   # Right mouth corner
        ])
        
        return landmarks

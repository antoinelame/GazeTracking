"""
Performance monitoring module for FPS tracking, latency measurement,
and distance validation.
"""

import time
from collections import deque
from typing import Optional, Tuple


class PerformanceMonitor:
    """
    Monitors system performance including FPS, latency, and distance validation.
    """
    
    def __init__(self,
                 target_fps: float = 100.0,
                 min_fps: float = 50.0,
                 distance_range: Tuple[float, float] = (20.0, 30.0),  # inches
                 fps_window_size: int = 30):
        """
        Initialize performance monitor.
        
        Args:
            target_fps: Target FPS (default: 100Hz)
            min_fps: Minimum acceptable FPS (default: 50Hz)
            distance_range: Valid distance range in inches (min, max)
            fps_window_size: Number of frames to average for FPS calculation
        """
        self.target_fps = target_fps
        self.min_fps = min_fps
        self.distance_range = distance_range
        
        # FPS tracking
        self.fps_window_size = fps_window_size
        self.frame_times = deque(maxlen=fps_window_size)
        self.current_fps = 0.0
        self.frame_count = 0
        self.last_frame_time = None
        
        # Latency tracking
        self.processing_times = deque(maxlen=fps_window_size)
        self.current_latency_ms = 0.0
        
        # Distance estimation
        self.current_distance = None
        self.distance_valid = False
        
        # Performance warnings
        self.low_fps_warning = False
        self.high_latency_warning = False
        self.distance_warning = False
    
    def start_frame(self) -> float:
        """
        Mark the start of frame processing.
        
        Returns:
            Timestamp of frame start
        """
        frame_start = time.time()
        self.last_frame_time = frame_start
        return frame_start
    
    def end_frame(self, frame_start: float):
        """
        Mark the end of frame processing and update metrics.
        
        Args:
            frame_start: Timestamp from start_frame()
        """
        frame_end = time.time()
        
        # Calculate processing latency
        processing_time = (frame_end - frame_start) * 1000.0  # Convert to ms
        self.processing_times.append(processing_time)
        self.current_latency_ms = sum(self.processing_times) / len(self.processing_times)
        
        # Calculate FPS
        if self.last_frame_time is not None:
            frame_interval = frame_end - self.last_frame_time
            if frame_interval > 0:
                frame_fps = 1.0 / frame_interval
                self.frame_times.append(frame_fps)
                
                if len(self.frame_times) > 0:
                    self.current_fps = sum(self.frame_times) / len(self.frame_times)
        
        self.frame_count += 1
        
        # Check performance warnings
        self._check_warnings()
    
    def _check_warnings(self):
        """Check if performance is below thresholds"""
        self.low_fps_warning = self.current_fps < self.min_fps
        self.high_latency_warning = self.current_latency_ms > (1000.0 / self.min_fps)
        self.distance_warning = not self.distance_valid
    
    def update_distance(self, face_bbox: Optional[Tuple[int, int, int, int]], 
                       frame_width: int, frame_height: int):
        """
        Estimate distance from camera based on face size.
        
        Args:
            face_bbox: Face bounding box (x, y, width, height) or None
            frame_width: Width of the frame
            frame_height: Height of the frame
        """
        if face_bbox is None:
            self.current_distance = None
            self.distance_valid = False
            return
        
        x, y, w, h = face_bbox
        
        # Estimate distance based on face size
        # This is a simplified model - can be calibrated for specific cameras
        # Assumes: face at 25 inches = ~15% of frame width
        face_width_ratio = w / frame_width if frame_width > 0 else 0
        face_height_ratio = h / frame_height if frame_height > 0 else 0
        face_size_ratio = (face_width_ratio + face_height_ratio) / 2.0
        
        # Rough estimation: distance is inversely proportional to face size
        # Calibration constant (needs to be tuned for specific setup)
        reference_distance = 25.0  # inches
        reference_size = 0.15  # 15% of frame
        
        if face_size_ratio > 0:
            estimated_distance = (reference_distance * reference_size) / face_size_ratio
            self.current_distance = estimated_distance
            
            # Check if distance is in valid range
            min_dist, max_dist = self.distance_range
            self.distance_valid = min_dist <= estimated_distance <= max_dist
        else:
            self.current_distance = None
            self.distance_valid = False
        
        self._check_warnings()
    
    def get_fps(self) -> float:
        """Get current FPS"""
        return self.current_fps
    
    def get_latency_ms(self) -> float:
        """Get current processing latency in milliseconds"""
        return self.current_latency_ms
    
    def get_distance(self) -> Optional[float]:
        """Get estimated distance in inches"""
        return self.current_distance
    
    def is_distance_valid(self) -> bool:
        """Check if distance is within valid range"""
        return self.distance_valid
    
    def get_performance_status(self) -> dict:
        """
        Get comprehensive performance status.
        
        Returns:
            Dictionary with performance metrics and warnings
        """
        return {
            'fps': self.current_fps,
            'target_fps': self.target_fps,
            'min_fps': self.min_fps,
            'latency_ms': self.current_latency_ms,
            'distance_inches': self.current_distance,
            'distance_valid': self.distance_valid,
            'distance_range': self.distance_range,
            'low_fps_warning': self.low_fps_warning,
            'high_latency_warning': self.high_latency_warning,
            'distance_warning': self.distance_warning,
            'frame_count': self.frame_count,
        }
    
    def is_performance_acceptable(self) -> bool:
        """
        Check if performance meets minimum requirements.
        
        Returns:
            True if performance is acceptable
        """
        return (not self.low_fps_warning and 
                not self.high_latency_warning and 
                self.distance_valid)
    
    def reset(self):
        """Reset performance monitor"""
        self.frame_times.clear()
        self.processing_times.clear()
        self.current_fps = 0.0
        self.current_latency_ms = 0.0
        self.frame_count = 0
        self.last_frame_time = None
        self.current_distance = None
        self.distance_valid = False
        self.low_fps_warning = False
        self.high_latency_warning = False
        self.distance_warning = False

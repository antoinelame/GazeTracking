"""
Data logging module for CSV export with high-precision timestamps.
Thread-safe logging for real-time performance.
"""

import csv
import time
import threading
from datetime import datetime
from typing import Optional, Dict, List
from collections import deque
import os


class DataLogger:
    """
    Thread-safe data logger for gaze tracking metrics.
    Exports data to CSV with high-precision timestamps.
    """
    
    def __init__(self, 
                 output_file: Optional[str] = None,
                 buffer_size: int = 100,
                 auto_flush: bool = True):
        """
        Initialize data logger.
        
        Args:
            output_file: Path to output CSV file. If None, auto-generates filename.
            buffer_size: Number of records to buffer before writing
            auto_flush: Automatically flush buffer when full
        """
        self.output_file = output_file or self._generate_filename()
        self.buffer_size = buffer_size
        self.auto_flush = auto_flush
        
        self.buffer = deque(maxlen=buffer_size)
        self.lock = threading.Lock()
        self.is_logging = False
        self.start_time = None
        self.record_count = 0
        
        # CSV headers
        self.headers = [
            'timestamp',
            'tracker_method',
            'left_pupil_x',
            'left_pupil_y',
            'right_pupil_x',
            'right_pupil_y',
            'left_pupil_diameter',
            'right_pupil_diameter',
            'gaze_angle_horizontal',
            'gaze_angle_vertical',
            'eye_state',
            'drowsiness_score',
            'fps',
            'face_detected',
            'processing_latency_ms'
        ]
        
        # Initialize CSV file
        self._initialize_csv()
    
    def _generate_filename(self) -> str:
        """Generate filename with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"gaze_tracking_data_{timestamp}.csv"
    
    def _initialize_csv(self):
        """Initialize CSV file with headers"""
        try:
            with open(self.output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
        except Exception as e:
            print(f"Error initializing CSV file: {e}")
    
    def start_logging(self):
        """Start logging session"""
        with self.lock:
            self.is_logging = True
            self.start_time = time.time()
            self.record_count = 0
    
    def stop_logging(self):
        """Stop logging session and flush buffer"""
        with self.lock:
            self.is_logging = False
            self.flush()
    
    def log(self, 
            tracker_method: str,
            left_pupil_coords: Optional[tuple] = None,
            right_pupil_coords: Optional[tuple] = None,
            left_pupil_diameter: Optional[float] = None,
            right_pupil_diameter: Optional[float] = None,
            gaze_angle: Optional[tuple] = None,
            eye_state: Optional[int] = None,
            drowsiness_score: Optional[float] = None,
            fps: Optional[float] = None,
            face_detected: Optional[bool] = None,
            processing_latency_ms: Optional[float] = None,
            timestamp: Optional[float] = None):
        """
        Log a data record.
        
        Args:
            tracker_method: Tracker method name
            left_pupil_coords: (x, y) coordinates of left pupil
            right_pupil_coords: (x, y) coordinates of right pupil
            left_pupil_diameter: Diameter of left pupil in pixels
            right_pupil_diameter: Diameter of right pupil in pixels
            gaze_angle: (horizontal, vertical) gaze angle in degrees
            eye_state: 1 for open, 0 for closed
            drowsiness_score: Drowsiness score (0.0-1.0)
            fps: Current FPS
            face_detected: True if face detected
            processing_latency_ms: Processing latency in milliseconds
            timestamp: Optional timestamp (defaults to current time)
        """
        if not self.is_logging:
            return
        
        if timestamp is None:
            timestamp = time.time()
        
        # Format timestamp with microsecond precision
        timestamp_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")
        
        # Extract gaze angles
        gaze_h = None
        gaze_v = None
        if gaze_angle is not None:
            gaze_h, gaze_v = gaze_angle
        
        # Extract pupil coordinates
        left_x, left_y = (left_pupil_coords if left_pupil_coords else (None, None))
        right_x, right_y = (right_pupil_coords if right_pupil_coords else (None, None))
        
        # Create record
        record = [
            timestamp_str,
            tracker_method,
            left_x,
            left_y,
            right_x,
            right_y,
            left_pupil_diameter,
            right_pupil_diameter,
            gaze_h,
            gaze_v,
            eye_state,
            drowsiness_score,
            fps,
            face_detected,
            processing_latency_ms
        ]
        
        # Thread-safe append to buffer
        with self.lock:
            self.buffer.append(record)
            self.record_count += 1
            
            # Auto-flush if buffer is full
            if self.auto_flush and len(self.buffer) >= self.buffer_size:
                self._flush_buffer()
    
    def _flush_buffer(self):
        """Flush buffer to CSV file (internal, assumes lock is held)"""
        if len(self.buffer) == 0:
            return
        
        try:
            with open(self.output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                while self.buffer:
                    writer.writerow(self.buffer.popleft())
        except Exception as e:
            print(f"Error writing to CSV file: {e}")
    
    def flush(self):
        """Flush buffer to CSV file (thread-safe)"""
        with self.lock:
            self._flush_buffer()
    
    def get_record_count(self) -> int:
        """Get number of records logged"""
        with self.lock:
            return self.record_count
    
    def get_output_file(self) -> str:
        """Get output file path"""
        return self.output_file
    
    def export_to_csv(self, output_path: Optional[str] = None):
        """
        Export all buffered data to CSV file.
        
        Args:
            output_path: Optional output path (defaults to current output_file)
        """
        if output_path is None:
            output_path = self.output_file
        
        self.flush()
        
        # If different path, copy file
        if output_path != self.output_file:
            import shutil
            try:
                shutil.copy2(self.output_file, output_path)
            except Exception as e:
                print(f"Error copying file: {e}")

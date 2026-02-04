"""
GUI application for real-time gaze tracking.
Uses tkinter for cross-platform compatibility.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
from gaze_tracking import GazeTracking
from gaze_tracking.safety_monitor import SafetyMonitor
from gaze_tracking.data_logger import DataLogger
from gaze_tracking.performance_monitor import PerformanceMonitor
import config


class GazeTrackingGUI:
    """
    Main GUI application for gaze tracking.
    """
    
    def __init__(self, root):
        """
        Initialize GUI.
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("Gaze Tracking System")
        self.root.geometry("1200x800")
        
        # Initialize components
        self.gaze = None
        self.safety_monitor = None
        self.data_logger = None
        self.performance_monitor = None
        
        # Video capture
        self.cap = None
        self.is_running = False
        self.is_recording = False
        
        # GUI update
        self.update_thread = None
        
        # Create GUI
        self._create_widgets()
        
        # Initialize system
        self._initialize_system()
        
        # Auto-start webcam after a short delay (allows GUI to fully render)
        self.root.after(500, self._auto_start_webcam)
    
    def _create_widgets(self):
        """Create GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Left panel: Video display
        video_frame = ttk.LabelFrame(main_frame, text="Video Feed", padding="5")
        video_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        main_frame.rowconfigure(0, weight=1)
        
        self.video_label = ttk.Label(video_frame, text="No video feed")
        self.video_label.pack(expand=True, fill=tk.BOTH)
        
        # Right panel: Controls and metrics
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        right_panel.columnconfigure(0, weight=1)
        
        # Controls frame
        controls_frame = ttk.LabelFrame(right_panel, text="Controls", padding="10")
        controls_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        controls_frame.columnconfigure(1, weight=1)
        
        # Start/Stop button (tracker selection removed - using default)
        self.start_stop_btn = ttk.Button(controls_frame, text="Start Tracking", 
                                         command=self._toggle_tracking)
        self.start_stop_btn.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Record button
        self.record_btn = ttk.Button(controls_frame, text="Start Recording", 
                                     command=self._toggle_recording, state=tk.DISABLED)
        self.record_btn.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Export button
        self.export_btn = ttk.Button(controls_frame, text="Export Data", 
                                     command=self._export_data, state=tk.DISABLED)
        self.export_btn.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Metrics frame
        metrics_frame = ttk.LabelFrame(right_panel, text="Metrics", padding="10")
        metrics_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        right_panel.rowconfigure(1, weight=1)
        metrics_frame.columnconfigure(1, weight=1)
        
        # Create metrics labels
        self.metrics_labels = {}
        metrics = [
            ("FPS:", "fps"),
            ("Latency (ms):", "latency"),
            ("Distance (in):", "distance"),
            ("Pupil Diameter (px):", "pupil_diameter"),
            ("Left Eye Center:", "left_eye_center"),
            ("Right Eye Center:", "right_eye_center"),
            ("Gaze Angle H:", "gaze_h"),
            ("Gaze Angle V:", "gaze_v"),
            ("Eye State:", "eye_state"),
            ("Drowsiness Score:", "drowsiness"),
            ("Face Detected:", "face_detected"),
        ]
        
        for i, (label_text, key) in enumerate(metrics):
            ttk.Label(metrics_frame, text=label_text).grid(row=i, column=0, sticky=tk.W, pady=2)
            value_label = ttk.Label(metrics_frame, text="N/A", foreground="blue")
            value_label.grid(row=i, column=1, sticky=tk.W, padx=10)
            self.metrics_labels[key] = value_label
        
        # Status frame
        status_frame = ttk.LabelFrame(right_panel, text="Status", padding="10")
        status_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.status_label = ttk.Label(status_frame, text="Ready", foreground="green")
        self.status_label.pack()
        
        # Alarms frame
        alarms_frame = ttk.LabelFrame(right_panel, text="Alarms", padding="10")
        alarms_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.alarm_label = ttk.Label(alarms_frame, text="No alarms", foreground="green")
        self.alarm_label.pack()
    
    def _initialize_system(self):
        """Initialize gaze tracking system"""
        try:
            self.gaze = GazeTracking(tracker_type=config.DEFAULT_TRACKER)
            self.safety_monitor = SafetyMonitor(
                out_of_frame_threshold=config.OUT_OF_FRAME_THRESHOLD,
                perclos_threshold=config.PERCLOS_THRESHOLD,
                enable_audio=config.ENABLE_AUDIO_ALARMS,
                enable_visual=config.ENABLE_VISUAL_ALARMS
            )
            self.data_logger = DataLogger(
                buffer_size=config.DATA_LOG_BUFFER_SIZE,
                auto_flush=config.DATA_LOG_AUTO_FLUSH
            )
            self.performance_monitor = PerformanceMonitor(
                target_fps=config.TARGET_FPS,
                min_fps=config.MIN_FPS,
                distance_range=config.DISTANCE_RANGE_INCHES
            )
            self.status_label.config(text="System initialized - Starting webcam...", foreground="green")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize system: {e}")
            self.status_label.config(text="Initialization failed", foreground="red")
    
    def _auto_start_webcam(self):
        """Automatically start webcam when GUI opens"""
        if not self.is_running:
            self._start_tracking()
    
    
    def _toggle_tracking(self):
        """Toggle video tracking"""
        if not self.is_running:
            self._start_tracking()
        else:
            self._stop_tracking()
    
    def _start_tracking(self):
        """Start video tracking"""
        try:
            self.cap = cv2.VideoCapture(config.WEBCAM_INDEX)
            if not self.cap.isOpened():
                raise Exception("Could not open webcam")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.WEBCAM_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.WEBCAM_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, config.WEBCAM_FPS)
            
            self.is_running = True
            self.start_stop_btn.config(text="Stop Tracking")
            self.record_btn.config(state=tk.NORMAL)
            self.status_label.config(text="Tracking active", foreground="green")
            
            # Start update thread
            self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.update_thread.start()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start tracking: {e}")
            self.status_label.config(text="Start failed", foreground="red")
    
    def _stop_tracking(self):
        """Stop video tracking"""
        self.is_running = False
        
        if self.is_recording:
            self._toggle_recording()
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        self.start_stop_btn.config(text="Start Tracking")
        self.record_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Tracking stopped", foreground="orange")
        self.video_label.config(image='', text="No video feed")
    
    def _toggle_recording(self):
        """Toggle data recording"""
        if not self.is_recording:
            self.data_logger.start_logging()
            self.is_recording = True
            self.record_btn.config(text="Stop Recording")
            self.export_btn.config(state=tk.DISABLED)
            self.status_label.config(text="Recording...", foreground="red")
        else:
            self.data_logger.stop_logging()
            self.is_recording = False
            self.record_btn.config(text="Start Recording")
            self.export_btn.config(state=tk.NORMAL)
            self.status_label.config(text="Recording stopped", foreground="orange")
    
    def _export_data(self):
        """Export recorded data"""
        if self.data_logger.get_record_count() == 0:
            messagebox.showinfo("Info", "No data to export")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.data_logger.export_to_csv(filename)
                messagebox.showinfo("Success", f"Data exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export data: {e}")
    
    def _update_loop(self):
        """Main update loop for video processing"""
        while self.is_running:
            frame_start = self.performance_monitor.start_frame()
            
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Process frame
            self.gaze.refresh(frame)
            
            # Update performance monitor
            frame_height, frame_width = frame.shape[:2]
            self.performance_monitor.update_distance(
                self.gaze.face_bbox, frame_width, frame_height
            )
            self.performance_monitor.end_frame(frame_start)
            
            # Update safety monitor
            eye_state = self.gaze.eye_state()
            self.safety_monitor.update(
                self.gaze.is_face_detected(),
                eye_state,
                time.time()
            )
            
            # Log data if recording
            if self.is_recording:
                self.data_logger.log(
                    tracker_method=self.gaze.tracker_type,
                    left_pupil_coords=self.gaze.pupil_left_coords(),
                    right_pupil_coords=self.gaze.pupil_right_coords(),
                    left_pupil_diameter=self.gaze.pupil_left_diameter(),
                    right_pupil_diameter=self.gaze.pupil_right_diameter(),
                    gaze_angle=self.gaze.gaze_angle(),
                    eye_state=eye_state,
                    drowsiness_score=self.safety_monitor.drowsiness_monitor.get_drowsiness_score(),
                    fps=self.performance_monitor.get_fps(),
                    face_detected=self.gaze.is_face_detected(),
                    processing_latency_ms=self.performance_monitor.get_latency_ms()
                )
            
            # Update GUI (throttled)
            if config.GUI_UPDATE_RATE > 0:
                time.sleep(1.0 / config.GUI_UPDATE_RATE)
            
            # Schedule GUI update on main thread
            self.root.after(0, self._update_gui, frame)
    
    def _update_gui(self, frame):
        """Update GUI elements"""
        if not self.is_running:
            return
        
        # Update video display
        if config.GUI_SHOW_ANNOTATIONS:
            annotated_frame = self.gaze.annotated_frame()
            if annotated_frame is not None:
                frame = annotated_frame
        
        # Convert frame for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frame_tk = ImageTk.PhotoImage(image=frame_pil)
        
        self.video_label.config(image=frame_tk, text="")
        self.video_label.image = frame_tk  # Keep a reference
        
        # Update metrics
        if config.GUI_SHOW_METRICS:
            self.metrics_labels["fps"].config(
                text=f"{self.performance_monitor.get_fps():.1f}"
            )
            self.metrics_labels["latency"].config(
                text=f"{self.performance_monitor.get_latency_ms():.2f}"
            )
            distance = self.performance_monitor.get_distance()
            self.metrics_labels["distance"].config(
                text=f"{distance:.1f}" if distance else "N/A"
            )
            diameter = self.gaze.pupil_diameter()
            self.metrics_labels["pupil_diameter"].config(
                text=f"{diameter:.1f}" if diameter else "N/A"
            )
            
            # Eye center coordinates
            left_eye_center = self.gaze.eye_left_center()
            if left_eye_center:
                self.metrics_labels["left_eye_center"].config(
                    text=f"({left_eye_center[0]}, {left_eye_center[1]})"
                )
            else:
                self.metrics_labels["left_eye_center"].config(text="N/A")
            
            right_eye_center = self.gaze.eye_right_center()
            if right_eye_center:
                self.metrics_labels["right_eye_center"].config(
                    text=f"({right_eye_center[0]}, {right_eye_center[1]})"
                )
            else:
                self.metrics_labels["right_eye_center"].config(text="N/A")
            
            gaze_angle = self.gaze.gaze_angle()
            if gaze_angle:
                self.metrics_labels["gaze_h"].config(text=f"{gaze_angle[0]:.1f}°")
                self.metrics_labels["gaze_v"].config(text=f"{gaze_angle[1]:.1f}°")
            else:
                self.metrics_labels["gaze_h"].config(text="N/A")
                self.metrics_labels["gaze_v"].config(text="N/A")
            
            eye_state = "Open" if self.gaze.eye_state() == 1 else "Closed"
            self.metrics_labels["eye_state"].config(text=eye_state)
            
            drowsiness = self.safety_monitor.drowsiness_monitor.get_drowsiness_score()
            self.metrics_labels["drowsiness"].config(text=f"{drowsiness:.2f}")
            
            face_detected = "Yes" if self.gaze.is_face_detected() else "No"
            self.metrics_labels["face_detected"].config(text=face_detected)
        
        # Update alarms with different colors and messages
        status = self.safety_monitor.get_status()
        if status['out_of_frame_alarm']:
            self.alarm_label.config(text="⚠️ OUT OF FRAME - Participant not visible!", foreground="orange")
        elif status['drowsiness_alarm']:
            self.alarm_label.config(text="⚠️ DROWSINESS DETECTED - Alert participant!", foreground="red")
        else:
            self.alarm_label.config(text="✓ No alarms", foreground="green")
    
    def on_closing(self):
        """Handle window closing"""
        if self.is_running:
            self._stop_tracking()
        self.root.destroy()


def main():
    """Main entry point for GUI application"""
    root = tk.Tk()
    app = GazeTrackingGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()

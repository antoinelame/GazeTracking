"""
Safety monitoring module for out-of-frame detection and drowsiness monitoring.
Includes alarm system with audio and visual notifications.
"""

import time
import threading
from collections import deque
from typing import Optional, Callable


class OutOfFrameMonitor:
    """
    Monitors if the participant exits the camera's field of view.
    Triggers alarm after consecutive frames without face detection.
    """
    
    def __init__(self, threshold_frames=5, alarm_callback=None):
        """
        Initialize out-of-frame monitor.
        
        Args:
            threshold_frames: Number of consecutive frames without face to trigger alarm
            alarm_callback: Callback function called when alarm is triggered
        """
        self.threshold_frames = threshold_frames
        self.alarm_callback = alarm_callback
        self.consecutive_no_face = 0
        self.alarm_active = False
        self.last_face_time = None
    
    def update(self, face_detected: bool):
        """
        Update monitor with current frame's face detection status.
        
        Args:
            face_detected: True if face detected in current frame
        """
        if face_detected:
            self.consecutive_no_face = 0
            self.alarm_active = False
            self.last_face_time = time.time()
        else:
            self.consecutive_no_face += 1
            
            if self.consecutive_no_face >= self.threshold_frames and not self.alarm_active:
                self.alarm_active = True
                if self.alarm_callback:
                    self.alarm_callback("out_of_frame")
    
    def is_alarm_active(self) -> bool:
        """Check if alarm is currently active"""
        return self.alarm_active
    
    def reset(self):
        """Reset the monitor"""
        self.consecutive_no_face = 0
        self.alarm_active = False
        self.last_face_time = None


class DrowsinessMonitor:
    """
    Monitors drowsiness using PERCLOS (Percentage of Eyelid Closure)
    and blink frequency analysis.
    """
    
    def __init__(self, 
                 perclos_threshold=0.5,
                 blink_frequency_threshold=0.1,  # blinks per second
                 window_size=60,  # frames (assuming ~50-100Hz)
                 alarm_callback=None):
        """
        Initialize drowsiness monitor.
        
        Args:
            perclos_threshold: PERCLOS threshold (0.0-1.0) to trigger alarm
            blink_frequency_threshold: Minimum blinks per second (too low = drowsy)
            window_size: Number of frames to analyze for PERCLOS
            alarm_callback: Callback function called when drowsiness detected
        """
        self.perclos_threshold = perclos_threshold
        self.blink_frequency_threshold = blink_frequency_threshold
        self.window_size = window_size
        self.alarm_callback = alarm_callback
        
        # Store eye states in a sliding window
        self.eye_states = deque(maxlen=window_size)
        self.blink_times = deque(maxlen=100)  # Store last 100 blink times
        self.last_blink_time = None
        
        self.alarm_active = False
        self.drowsiness_score = 0.0
    
    def update(self, eye_state: int, timestamp: Optional[float] = None):
        """
        Update monitor with current eye state.
        
        Args:
            eye_state: 1 for open, 0 for closed
            timestamp: Optional timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Update eye state window
        self.eye_states.append(eye_state)
        
        # Detect blink (transition from open to closed to open)
        if len(self.eye_states) >= 3:
            prev_prev = self.eye_states[-3]
            prev = self.eye_states[-2]
            curr = self.eye_states[-1]
            
            # Blink detected: open -> closed -> open
            if prev_prev == 1 and prev == 0 and curr == 1:
                self.blink_times.append(timestamp)
                self.last_blink_time = timestamp
        
        # Calculate PERCLOS (percentage of time eyes are closed)
        if len(self.eye_states) >= self.window_size:
            closed_count = sum(1 for state in self.eye_states if state == 0)
            perclos = closed_count / len(self.eye_states)
            self.drowsiness_score = perclos
            
            # Check for drowsiness
            if perclos >= self.perclos_threshold and not self.alarm_active:
                self.alarm_active = True
                if self.alarm_callback:
                    self.alarm_callback("drowsiness")
        else:
            # Not enough data yet, use current state
            if len(self.eye_states) > 0:
                closed_ratio = sum(1 for state in self.eye_states if state == 0) / len(self.eye_states)
                self.drowsiness_score = closed_ratio
    
    def get_blink_frequency(self) -> float:
        """
        Calculate current blink frequency (blinks per second).
        
        Returns:
            Blink frequency in blinks per second
        """
        if len(self.blink_times) < 2:
            return 0.0
        
        # Calculate time span
        time_span = self.blink_times[-1] - self.blink_times[0]
        if time_span <= 0:
            return 0.0
        
        # Calculate frequency
        frequency = (len(self.blink_times) - 1) / time_span
        return frequency
    
    def get_drowsiness_score(self) -> float:
        """
        Get current drowsiness score (0.0 = alert, 1.0 = very drowsy).
        
        Returns:
            Drowsiness score (0.0-1.0)
        """
        return self.drowsiness_score
    
    def is_alarm_active(self) -> bool:
        """Check if drowsiness alarm is currently active"""
        return self.alarm_active
    
    def reset(self):
        """Reset the monitor"""
        self.eye_states.clear()
        self.blink_times.clear()
        self.alarm_active = False
        self.drowsiness_score = 0.0
        self.last_blink_time = None


class AlarmSystem:
    """
    Alarm system for audio and visual notifications.
    """
    
    def __init__(self, enable_audio=True, enable_visual=True):
        """
        Initialize alarm system.
        
        Args:
            enable_audio: Enable audio alarms
            enable_visual: Enable visual alarms
        """
        self.enable_audio = enable_audio
        self.enable_visual = enable_visual
        self.audio_player = None
        
        # Try to initialize audio
        if self.enable_audio:
            try:
                import pygame
                pygame.mixer.init()
                self.audio_player = pygame
            except ImportError:
                try:
                    from playsound import playsound
                    self.audio_player = 'playsound'
                except ImportError:
                    print("Warning: No audio library available. Audio alarms disabled.")
                    self.enable_audio = False
                    self.audio_player = None
    
    def trigger_alarm(self, alarm_type: str):
        """
        Trigger an alarm with different sounds for different alarm types.
        
        Args:
            alarm_type: Type of alarm ('out_of_frame', 'drowsiness')
        """
        if alarm_type == "out_of_frame":
            message = "WARNING: Participant out of frame!"
            alarm_frequency = 800  # Lower frequency for out of frame
            alarm_duration = 0.3
        elif alarm_type == "drowsiness":
            message = "WARNING: Drowsiness detected!"
            alarm_frequency = 1200  # Higher frequency for drowsiness
            alarm_duration = 0.5
        else:
            message = f"WARNING: {alarm_type}"
            alarm_frequency = 1000
            alarm_duration = 0.4
        
        # Visual alarm (print to console, can be extended to GUI)
        if self.enable_visual:
            print(f"\n{'='*50}")
            print(f"{message}")
            print(f"{'='*50}\n")
        
        # Audio alarm with different sounds
        if self.enable_audio and self.audio_player:
            self._play_audio_alarm(alarm_frequency, alarm_duration)
    
    def _play_audio_alarm(self, frequency=1000, duration=0.5):
        """
        Play audio alarm with specified frequency and duration.
        
        Args:
            frequency: Frequency in Hz (default: 1000)
            duration: Duration in seconds (default: 0.5)
        """
        try:
            if self.audio_player == 'playsound':
                # Generate a beep using system beep
                import os
                if os.name == 'nt':  # Windows
                    import winsound
                    winsound.Beep(int(frequency), int(duration * 1000))
                else:  # Unix/Mac
                    # Play multiple beeps for different alarm types
                    beep_count = 2 if frequency > 1000 else 1
                    for _ in range(beep_count):
                        os.system('printf "\a"')  # Bell character
                        time.sleep(0.2)
            elif hasattr(self.audio_player, 'mixer'):
                # pygame - generate a beep tone with specified frequency
                import numpy as np
                sample_rate = 44100
                
                t = np.linspace(0, duration, int(sample_rate * duration))
                # Add a slight fade in/out to avoid clicks
                fade_samples = int(sample_rate * 0.05)  # 50ms fade
                wave = np.sin(2 * np.pi * frequency * t)
                
                # Apply fade
                if len(wave) > fade_samples * 2:
                    wave[:fade_samples] *= np.linspace(0, 1, fade_samples)
                    wave[-fade_samples:] *= np.linspace(1, 0, fade_samples)
                
                wave = (wave * 32767).astype(np.int16)
                
                sound = self.audio_player.sndarray.make_sound(wave)
                sound.play()
        except Exception as e:
            print(f"Error playing audio alarm: {e}")


class SafetyMonitor:
    """
    Combined safety monitoring system.
    """
    
    def __init__(self, 
                 out_of_frame_threshold=5,
                 perclos_threshold=0.5,
                 enable_audio=True,
                 enable_visual=True):
        """
        Initialize safety monitor.
        
        Args:
            out_of_frame_threshold: Frames without face to trigger alarm
            perclos_threshold: PERCLOS threshold for drowsiness
            enable_audio: Enable audio alarms
            enable_visual: Enable visual alarms
        """
        self.alarm_system = AlarmSystem(enable_audio, enable_visual)
        
        self.out_of_frame_monitor = OutOfFrameMonitor(
            threshold_frames=out_of_frame_threshold,
            alarm_callback=self.alarm_system.trigger_alarm
        )
        
        self.drowsiness_monitor = DrowsinessMonitor(
            perclos_threshold=perclos_threshold,
            alarm_callback=self.alarm_system.trigger_alarm
        )
    
    def update(self, face_detected: bool, eye_state: Optional[int] = None, timestamp: Optional[float] = None):
        """
        Update safety monitors with current frame data.
        
        Args:
            face_detected: True if face detected
            eye_state: 1 for open, 0 for closed (optional)
            timestamp: Optional timestamp
        """
        self.out_of_frame_monitor.update(face_detected)
        
        if eye_state is not None:
            self.drowsiness_monitor.update(eye_state, timestamp)
    
    def get_status(self) -> dict:
        """
        Get current safety status.
        
        Returns:
            Dictionary with safety status information
        """
        return {
            'out_of_frame_alarm': self.out_of_frame_monitor.is_alarm_active(),
            'drowsiness_alarm': self.drowsiness_monitor.is_alarm_active(),
            'drowsiness_score': self.drowsiness_monitor.get_drowsiness_score(),
            'blink_frequency': self.drowsiness_monitor.get_blink_frequency(),
            'face_detected': self.out_of_frame_monitor.last_face_time is not None,
        }
    
    def reset(self):
        """Reset all monitors"""
        self.out_of_frame_monitor.reset()
        self.drowsiness_monitor.reset()

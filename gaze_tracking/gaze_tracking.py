import os
import cv2
import mediapipe as mp
from .eye import Eye
from .calibration import Calibration


class GazeTracking(object):
    """
    This class tracks the user's gaze.
    It provides useful information like the position of the eyes
    and pupils and allows to know if the eyes are open or closed
    """

    def __init__(self):
        self.frame: cv2.typing.MatLike | None = None
        self.eye_left: Eye | None = None
        self.eye_right: Eye | None = None
        self.calibration: Calibration = Calibration()

        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    @property
    def pupils_located(self) -> bool:
        """Check that the pupils have been located"""
        try:
            if (self.eye_left and self.eye_right and 
                self.eye_left.pupil and self.eye_right.pupil and
                self.eye_left.pupil.x is not None and self.eye_left.pupil.y is not None and
                self.eye_right.pupil.x is not None and self.eye_right.pupil.y is not None):
                int(self.eye_left.pupil.x)
                int(self.eye_left.pupil.y)
                int(self.eye_right.pupil.x)
                int(self.eye_right.pupil.y)
                return True
            return False
        except Exception:
            return False

    def _analyze(self) -> None:
        """Detects the face and initialize Eye objects"""
        if self.frame is None:
            return
            
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            # Store landmarks for iris detection
            self._last_landmarks = landmarks
            self.eye_left = Eye(self.frame, landmarks, 0, self.calibration)
            self.eye_right = Eye(self.frame, landmarks, 1, self.calibration)
        else:
            self.eye_left = None
            self.eye_right = None
            self._last_landmarks = None

    def refresh(self, frame: cv2.typing.MatLike) -> None:
        """Refreshes the frame and analyzes it.

        Arguments:
            frame (numpy.ndarray): The frame to analyze
        """
        self.frame = frame
        self._analyze()

    def pupil_left_coords(self) -> tuple[int, int] | None:
        """Returns the coordinates of the left pupil"""
        if self.pupils_located and self.eye_left and self.eye_left.pupil:
            try:
                x = self.eye_left.origin[0] + self.eye_left.pupil.x
                y = self.eye_left.origin[1] + self.eye_left.pupil.y
                return (x, y)
            except (TypeError, AttributeError):
                return None
        return None

    def pupil_right_coords(self) -> tuple[int, int] | None:
        """Returns the coordinates of the right pupil"""
        if self.pupils_located and self.eye_right and self.eye_right.pupil:
            try:
                x = self.eye_right.origin[0] + self.eye_right.pupil.x
                y = self.eye_right.origin[1] + self.eye_right.pupil.y
                return (x, y)
            except (TypeError, AttributeError):
                return None
        return None

    def horizontal_ratio(self) -> float | None:
        """Returns a number between 0.0 and 1.0 that indicates the
        horizontal direction of the gaze. The extreme right is 0.0,
        the center is 0.5 and the extreme left is 1.0
        """
        if self.pupils_located and self.eye_left and self.eye_right and hasattr(self, '_last_landmarks') and self._last_landmarks:
            try:
                landmarks = self._last_landmarks
                
                # Get the actual eye boundaries from MediaPipe landmarks
                # Left eye: points 33 (left corner) to 133 (right corner)
                # Right eye: points 362 (left corner) to 263 (right corner)
                
                # Left eye boundaries
                left_eye_left = landmarks.landmark[33].x   # Left corner of left eye
                left_eye_right = landmarks.landmark[133].x # Right corner of left eye
                
                # Right eye boundaries  
                right_eye_left = landmarks.landmark[362].x # Left corner of right eye
                right_eye_right = landmarks.landmark[263].x # Right corner of right eye
                
                # Get pupil positions in the original frame coordinates
                left_pupil_x = self.eye_left.origin[0] + self.eye_left.pupil.x
                right_pupil_x = self.eye_right.origin[0] + self.eye_right.pupil.x
                
                # Convert to normalized coordinates (0-1) relative to frame width
                frame_width = self.frame.shape[1]
                left_pupil_norm = left_pupil_x / frame_width
                right_pupil_norm = right_pupil_x / frame_width
                
                # Calculate ratios relative to actual eye boundaries
                left_eye_width = left_eye_right - left_eye_left
                right_eye_width = right_eye_right - right_eye_left
                
                if left_eye_width > 0 and right_eye_width > 0:
                    # Calculate how far pupils are from the left edge of each eye
                    left_ratio = (left_pupil_norm - left_eye_left) / left_eye_width
                    right_ratio = (right_pupil_norm - right_eye_left) / right_eye_width
                    
                    # Average both eyes
                    avg_ratio = (left_ratio + right_ratio) / 2
                    
                    # Clamp to 0-1 range
                    return max(0.0, min(1.0, avg_ratio))
                
            except (TypeError, ZeroDivisionError, AttributeError):
                return None
        return None

    def vertical_ratio(self) -> float | None:
        """Returns a number between 0.0 and 1.0 that indicates the
        vertical direction of the gaze. The extreme top is 0.0,
        the center is 0.5 and the extreme bottom is 1.0
        """
        if self.pupils_located and self.eye_left and self.eye_right:
            try:
                pupil_left = self.eye_left.pupil.y / (self.eye_left.center[1] * 2 - 10)
                pupil_right = self.eye_right.pupil.y / (self.eye_right.center[1] * 2 - 10)
                return (pupil_left + pupil_right) / 2
            except (TypeError, ZeroDivisionError):
                return None
        return None

    def is_right(self) -> bool | None:
        """Returns true if the user is looking to the right"""
        if self.pupils_located:
            ratio = self.horizontal_ratio()
            return ratio <= 0.4 if ratio is not None else None
        return None

    def is_left(self) -> bool | None:
        """Returns true if the user is looking to the left"""
        if self.pupils_located:
            ratio = self.horizontal_ratio()
            return ratio >= 0.6 if ratio is not None else None
        return None

    def is_center(self) -> bool | None:
        """Returns true if the user is looking to the center"""
        if self.pupils_located:
            ratio = self.horizontal_ratio()
            if ratio is not None:
                return 0.4 < ratio < 0.6
        return None

    def is_blinking(self) -> bool | None:
        """Returns true if the user closes his eyes - DISABLED for gaze direction only"""
        return None

    def annotated_frame(self) -> cv2.typing.MatLike:
        """Returns the main frame with pupils highlighted"""
        frame = self.frame.copy()

        if self.pupils_located:
            color = (0, 255, 0)
            left_coords = self.pupil_left_coords()
            right_coords = self.pupil_right_coords()
            
            if left_coords:
                x_left, y_left = left_coords
                cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
                cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            
            if right_coords:
                x_right, y_right = right_coords
                cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
                cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)

        return frame

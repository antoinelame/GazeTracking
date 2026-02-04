import numpy as np
import cv2


class Pupil(object):
    """
    This class detects the iris of an eye and estimates
    the position of the pupil and calculates its diameter.
    """

    def __init__(self, eye_frame, threshold):
        self.iris_frame = None
        self.threshold = threshold
        self.x = None
        self.y = None
        self.diameter = None  # Diameter in pixels

        self.detect_iris(eye_frame)

    @staticmethod
    def image_processing(eye_frame, threshold):
        """Performs operations on the eye frame to isolate the iris

        Arguments:
            eye_frame (numpy.ndarray): Frame containing an eye and nothing else
            threshold (int): Threshold value used to binarize the eye frame

        Returns:
            A frame with a single element representing the iris
        """
        kernel = np.ones((3, 3), np.uint8)
        new_frame = cv2.bilateralFilter(eye_frame, 10, 15, 15)
        new_frame = cv2.erode(new_frame, kernel, iterations=3)
        new_frame = cv2.threshold(new_frame, threshold, 255, cv2.THRESH_BINARY)[1]

        return new_frame

    def detect_iris(self, eye_frame):
        """Detects the iris and estimates the position of the iris by
        calculating the centroid. Also calculates the diameter.

        Arguments:
            eye_frame (numpy.ndarray): Frame containing an eye and nothing else
        """
        self.iris_frame = self.image_processing(eye_frame, self.threshold)

        contours, _ = cv2.findContours(self.iris_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        contours = sorted(contours, key=cv2.contourArea)

        try:
            # Get the largest contour (pupil)
            pupil_contour = contours[-2] if len(contours) >= 2 else contours[-1]
            
            # Calculate centroid
            moments = cv2.moments(pupil_contour)
            if moments['m00'] != 0:
                self.x = int(moments['m10'] / moments['m00'])
                self.y = int(moments['m01'] / moments['m00'])
                
                # Calculate diameter
                self.diameter = self.calculate_diameter(pupil_contour)
            else:
                self.x = None
                self.y = None
                self.diameter = None
        except (IndexError, ZeroDivisionError):
            self.x = None
            self.y = None
            self.diameter = None
    
    def calculate_diameter(self, contour):
        """
        Calculate pupil diameter from contour using multiple methods.
        
        Arguments:
            contour: Contour of the pupil
            
        Returns:
            Diameter in pixels (average of multiple methods for accuracy)
        """
        if contour is None or len(contour) < 5:
            return None
        
        # Method 1: Using minimum enclosing circle
        (_, _), radius = cv2.minEnclosingCircle(contour)
        diameter_circle = radius * 2
        
        # Method 2: Using bounding box
        x, y, w, h = cv2.boundingRect(contour)
        diameter_box = max(w, h)
        
        # Method 3: Using area (assuming circular pupil)
        area = cv2.contourArea(contour)
        if area > 0:
            diameter_area = 2 * np.sqrt(area / np.pi)
        else:
            diameter_area = 0
        
        # Method 4: Using fitEllipse (most accurate for elliptical pupils)
        try:
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                (_, _), (width, height), _ = ellipse
                diameter_ellipse = max(width, height)
            else:
                diameter_ellipse = diameter_circle
        except:
            diameter_ellipse = diameter_circle
        
        # Return average of all methods for robustness
        diameters = [d for d in [diameter_circle, diameter_box, diameter_area, diameter_ellipse] if d > 0]
        
        if len(diameters) > 0:
            return float(np.mean(diameters))
        
        return None

# Accuracy Improvement Guide

This document outlines various methods to improve the accuracy of the gaze tracking system.

## 1. Calibration

### Personal Calibration
The system automatically calibrates during the first 20 frames, but you can improve accuracy by:

- **Proper Positioning**: Maintain 20-30 inches distance from camera
- **Good Lighting**: Ensure even, front-facing lighting (avoid backlighting)
- **Stable Head Position**: Keep head relatively still during calibration
- **Extended Calibration**: Modify `CALIBRATION_FRAMES` in `config.py` to use more frames (e.g., 50-100)

### Manual Calibration
You can implement a calibration routine where the user looks at known points on screen:

```python
# Example: 9-point calibration
calibration_points = [
    (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),  # Top row
    (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),  # Middle row
    (0.1, 0.9), (0.5, 0.9), (0.9, 0.9),  # Bottom row
]
```

## 2. Better Face Detection Models

### Upgrade to More Accurate DNN Models

**Option A: Use MediaPipe Face Detection (Recommended)**
- More accurate than OpenCV DNN
- Better at handling various angles and lighting
- Install: `pip install mediapipe`
- Implementation: Create a new tracker `mediapipe_tracker.py`

**Option B: Use YOLOv8 Face Detection**
- State-of-the-art accuracy
- Install: `pip install ultralytics`
- Requires GPU for real-time performance

**Option C: Use RetinaFace**
- Excellent accuracy for face detection
- Install: `pip install retinaface`

### Current DNN Model Upgrade
Replace the current Caffe model with a better one:

1. Download newer models from OpenCV or other sources
2. Update `opencv_dnn_tracker.py` to use TensorFlow or ONNX models
3. Consider using MobileNet-SSD for better speed/accuracy balance

## 3. Better Eye Detection

### Use Facial Landmarks
Implement 68-point facial landmark detection:
- More precise eye region detection
- Better eye state classification
- Libraries: `dlib`, `mediapipe`, or `face_recognition`

### Deep Learning for Eye Detection
- Train or use pre-trained models for eye region detection
- Better handling of glasses, makeup, etc.

## 4. Improved Pupil Detection

### Better Preprocessing
```python
# Enhanced preprocessing in pupil.py
def enhanced_preprocessing(eye_frame):
    # Histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(eye_frame)
    
    # Bilateral filtering (already implemented, but can tune)
    filtered = cv2.bilateralFilter(enhanced, 15, 80, 80)
    
    # Adaptive thresholding with better parameters
    threshold = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 10
    )
    return threshold
```

### Multiple Detection Methods
Combine multiple pupil detection methods:
- Contour analysis (current)
- HoughCircles
- Template matching
- Deep learning-based pupil detection

### Temporal Smoothing
Apply Kalman filter or moving average to smooth pupil positions:
```python
from filterpy.kalman import KalmanFilter

# Smooth pupil coordinates over time
kf = KalmanFilter(dim_x=2, dim_z=2)
# Update with each new detection
```

## 5. Gaze Angle Calibration

### Screen-Based Calibration
Implement a calibration routine that maps pupil positions to screen coordinates:

1. Display calibration points on screen
2. User looks at each point
3. Record pupil positions
4. Create mapping function (polynomial regression or neural network)

### Distance Calibration
Calibrate for different distances:
- Measure actual distance
- Adjust gaze angle calculations based on distance
- Store calibration parameters per user

## 6. Hardware Improvements

### Better Camera
- Higher resolution (1080p or 4K)
- Higher frame rate (60fps+)
- Better low-light performance
- IR camera for better pupil detection

### Lighting
- Even, diffused lighting
- Avoid reflections on glasses
- Consider IR lighting for pupil detection

### Camera Position
- Mount camera at eye level
- Ensure stable mounting
- Minimize vibrations

## 7. Software Optimizations

### Higher Resolution Processing
Process at higher resolution, then downscale for display:
```python
# Process at 1280x720, display at 640x480
frame_high_res = cv2.resize(frame, (1280, 720))
# Process frame_high_res
```

### Multi-Scale Detection
Detect at multiple scales and combine results:
```python
scales = [0.8, 1.0, 1.2]
for scale in scales:
    scaled_frame = cv2.resize(frame, None, fx=scale, fy=scale)
    # Detect and combine results
```

### Region of Interest (ROI)
Focus processing on face/eye regions:
- Reduce processing area
- Increase resolution in ROI
- Faster processing = can use more sophisticated algorithms

## 8. Machine Learning Enhancements

### Train Custom Models
- Collect labeled data (pupil positions, gaze angles)
- Train CNN for pupil detection
- Train regression model for gaze angle prediction

### Transfer Learning
- Use pre-trained models (e.g., from gaze estimation datasets)
- Fine-tune on your specific setup

### Ensemble Methods
- Combine predictions from multiple models
- Weighted voting or averaging

## 9. Configuration Tuning

### Adjust Detection Parameters
In `config.py` and tracker files:
- `minNeighbors`: Lower = more detections (may have false positives)
- `scaleFactor`: Smaller = more thorough search (slower)
- `minSize`: Adjust based on expected face size
- `confidence_threshold`: For DNN models

### Calibration Parameters
- `CALIBRATION_FRAMES`: More frames = better calibration
- `PUPIL_DETECTION_THRESHOLD`: Tune for your lighting
- `GAZE_ANGLE_HORIZONTAL_RANGE`: Calibrate based on your setup

## 10. Validation and Testing

### Ground Truth Data
- Record sessions with known gaze targets
- Compare predictions to ground truth
- Calculate accuracy metrics (MAE, RMSE)

### A/B Testing
- Test different configurations
- Compare accuracy metrics
- Choose best configuration

## Quick Wins (Easiest to Implement)

1. **Increase calibration frames** to 50-100
2. **Improve lighting** in your environment
3. **Adjust camera position** to eye level
4. **Tune detection parameters** in config.py
5. **Use MediaPipe** for better face detection
6. **Implement temporal smoothing** for pupil positions
7. **Add histogram equalization** to preprocessing

## Recommended Implementation Order

1. **Week 1**: Improve lighting and positioning, increase calibration frames
2. **Week 2**: Implement MediaPipe face detection
3. **Week 3**: Add temporal smoothing and better preprocessing
4. **Week 4**: Implement screen-based calibration
5. **Ongoing**: Collect data, train custom models, iterate

## Resources

- [OpenCV Face Detection Models](https://github.com/opencv/opencv/tree/master/samples/dnn)
- [MediaPipe Face Detection](https://google.github.io/mediapipe/solutions/face_detection.html)
- [Gaze Estimation Datasets](https://github.com/swook/faze_preprocessor)
- [Kalman Filter Tutorial](https://filterpy.readthedocs.io/)

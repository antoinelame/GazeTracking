# Enhanced Gaze Tracking System

A modular, real-time eye tracking system using OpenCV with support for multiple tracking methods (ML and non-ML). Designed for clinical research applications with EEG/TEP/EMG data correlation.

## Features

- **Multiple Tracking Methods**: Support for OpenCV DNN (ML), Haar Cascade (non-ML), and Hybrid approaches
- **Real-time Performance**: Target 100Hz processing (minimum 50Hz)
- **Advanced Metrics**: Pupil diameter, gaze angle, eye state classification
- **Safety Monitoring**: Out-of-frame detection and drowsiness monitoring with alarms
- **Data Export**: CSV export with high-precision timestamps for correlation with other data
- **GUI Application**: User-friendly interface with real-time visualization
- **Modular Architecture**: Easy to extend with new tracking methods

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam
- OpenCV 4.8.0 or higher

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: `tkinter` is typically included with Python. If not available on your system, install it via your package manager:
- Ubuntu/Debian: `sudo apt-get install python3-tk`
- macOS: Usually included with Python
- Windows: Usually included with Python

### Verify Installation

```bash
python main.py --mode gui
```

## Quick Start

### GUI Application

Launch the GUI application:

```bash
python main.py
# or
python gui_app.py
```

### Basic Usage (Python API)

```python
import cv2
from gaze_tracking import GazeTracking

# Initialize with default tracker (DNN)
gaze = GazeTracking(tracker_type='dnn')

# Or use Haar Cascade (faster, less accurate)
# gaze = GazeTracking(tracker_type='haar')

# Or use Hybrid (best of both)
# gaze = GazeTracking(tracker_type='hybrid')

# Open webcam
webcam = cv2.VideoCapture(0)

while True:
    _, frame = webcam.read()
    gaze.refresh(frame)
    
    # Get metrics
    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    diameter = gaze.pupil_diameter()
    gaze_angle = gaze.gaze_angle()
    eye_state = gaze.eye_state()  # 1 = open, 0 = closed
    
    # Display annotated frame
    annotated = gaze.annotated_frame()
    cv2.imshow("Gaze Tracking", annotated)
    
    if cv2.waitKey(1) == 27:  # ESC key
        break

webcam.release()
cv2.destroyAllWindows()
```

## Tracker Methods Comparison

| Method | Type | Accuracy | Speed | Use Case |
|--------|------|----------|-------|----------|
| **OpenCV DNN** | ML | High | Medium | Default, best accuracy |
| **OpenCV Haar** | Non-ML | Medium | Fast | Fast processing, lower accuracy acceptable |
| **Hybrid** | Combined | High | Medium-Fast | Best balance, fallback support |

### Switching Trackers

You can switch trackers at runtime:

```python
# Switch to Haar tracker
gaze.switch_tracker('haar')

# Switch to Hybrid tracker
gaze.switch_tracker('hybrid')
```

## Architecture

### Modular Tracker System

The system uses a modular architecture with a base tracker interface:

```
gaze_tracking/
├── trackers/
│   ├── base_tracker.py       # Abstract base class
│   ├── opencv_dnn_tracker.py # ML-based tracker
│   ├── opencv_haar_tracker.py # Non-ML tracker
│   └── hybrid_tracker.py     # Combined approach
├── gaze_tracking.py          # Main API
├── eye.py                    # Eye detection
├── pupil.py                  # Pupil detection with diameter
├── safety_monitor.py         # Safety features
├── data_logger.py            # CSV export
└── performance_monitor.py    # Performance tracking
```

## API Reference

### GazeTracking Class

#### Initialization

```python
GazeTracking(tracker_type='dnn', tracker=None)
```

- `tracker_type`: 'dnn', 'haar', or 'hybrid' (default: 'dnn')
- `tracker`: Optional pre-initialized tracker instance

#### Methods

- `refresh(frame)`: Process a new frame
- `pupil_left_coords()`: Get left pupil coordinates (x, y)
- `pupil_right_coords()`: Get right pupil coordinates (x, y)
- `pupil_left_diameter()`: Get left pupil diameter in pixels
- `pupil_right_diameter()`: Get right pupil diameter in pixels
- `pupil_diameter()`: Get average pupil diameter
- `gaze_angle()`: Get gaze angle (horizontal, vertical) in degrees
- `horizontal_ratio()`: Get horizontal gaze ratio (0.0-1.0)
- `vertical_ratio()`: Get vertical gaze ratio (0.0-1.0)
- `eye_state()`: Get eye state (1 = open, 0 = closed)
- `is_blinking()`: Check if eyes are closed
- `is_face_detected()`: Check if face is detected
- `is_left()`, `is_right()`, `is_center()`: Gaze direction
- `annotated_frame()`: Get frame with annotations
- `switch_tracker(tracker_type)`: Switch tracker method

## Data Export Format

CSV files contain the following columns:

- `timestamp`: High-precision timestamp (microsecond accuracy)
- `tracker_method`: Tracker method used ('dnn', 'haar', 'hybrid')
- `left_pupil_x`, `left_pupil_y`: Left pupil coordinates
- `right_pupil_x`, `right_pupil_y`: Right pupil coordinates
- `left_pupil_diameter`, `right_pupil_diameter`: Pupil diameters in pixels
- `gaze_angle_horizontal`, `gaze_angle_vertical`: Gaze angles in degrees
- `eye_state`: 1 for open, 0 for closed
- `drowsiness_score`: Drowsiness score (0.0-1.0)
- `fps`: Current FPS
- `face_detected`: Boolean (True/False)
- `processing_latency_ms`: Processing latency in milliseconds

## Configuration

Edit `config.py` to customize:

- Tracker method (default: 'dnn')
- Performance targets (FPS, latency)
- Distance validation range (20-30 inches)
- Safety monitor thresholds
- Alarm settings
- Data logging options

## Safety Features

### Out-of-Frame Detection

Monitors if the participant exits the camera's field of view. Triggers alarm after configurable number of consecutive frames without face detection.

### Drowsiness Monitoring

Uses PERCLOS (Percentage of Eyelid Closure) and blink frequency analysis to detect drowsiness. Triggers alertness protocol when drowsiness is detected.

### Alarms

- **Audio**: System beep/alarm (requires pygame or playsound)
- **Visual**: On-screen warnings and console messages

## Performance Optimization

### Achieving Target FPS (100Hz)

1. **Use appropriate tracker**: Haar is faster but less accurate
2. **Reduce frame resolution**: Lower webcam resolution
3. **Optimize processing**: Disable unnecessary features
4. **Hardware**: Use a fast CPU/GPU

### Distance Validation

The system estimates distance from camera based on face size. Ensure participant is positioned 20-30 inches from camera for optimal results.

## Calibration

The system automatically calibrates pupil detection thresholds during the first 20 frames. For best results:

1. Position participant 20-30 inches from camera
2. Ensure good lighting
3. Keep face centered in frame
4. Wait for calibration to complete (first 20 frames)

## Troubleshooting

### Low FPS

- Try Haar tracker instead of DNN
- Reduce webcam resolution
- Close other applications
- Check CPU usage

### Poor Detection

- Ensure good lighting
- Position participant 20-30 inches from camera
- Check camera focus
- Try different tracker method

### No Face Detected

- Check camera connection
- Ensure participant is in frame
- Adjust lighting
- Check camera permissions

### Audio Alarms Not Working

- Install pygame: `pip install pygame`
- Or install playsound: `pip install playsound`
- Check system audio settings

## Integration with EEG/TEP/EMG Data

The CSV export format is designed for easy correlation with other physiological data:

1. Export gaze tracking data to CSV
2. Use timestamps to align with EEG/TEP/EMG data
3. All timestamps use microsecond precision for accurate synchronization

Example alignment:

```python
import pandas as pd

# Load gaze data
gaze_data = pd.read_csv('gaze_tracking_data.csv')
gaze_data['timestamp'] = pd.to_datetime(gaze_data['timestamp'])

# Load EEG data
eeg_data = pd.read_csv('eeg_data.csv')
eeg_data['timestamp'] = pd.to_datetime(eeg_data['timestamp'])

# Merge on timestamp
merged = pd.merge_asof(gaze_data, eeg_data, on='timestamp', direction='nearest')
```

## Model Files

OpenCV DNN tracker may require model files. The system will:
1. Try to use built-in OpenCV models
2. Fall back to Haar Cascade if DNN models not available
3. Use region-based detection if cascades fail

For best DNN performance, download OpenCV face detection models and place in `gaze_tracking/trained_models/opencv_dnn/`.

## License

This project is based on the original GazeTracking library by Antoine Lamé (MIT License). Enhanced version maintains MIT License.

## Contributing

Contributions welcome! Areas for improvement:
- Additional tracker methods
- Performance optimizations
- Calibration improvements
- Documentation enhancements

## References

- Original GazeTracking: https://github.com/antoinelame/GazeTracking
- OpenCV Documentation: https://docs.opencv.org/
- Meeting Minutes: January 29, 2026 - Eye Tracking Project Requirements

## Support

For issues or questions, please refer to the troubleshooting section or open an issue on the repository.

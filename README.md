# Enhanced Gaze Tracking System

A modular, real-time eye tracking system using OpenCV with support for multiple tracking methods (ML and non-ML). Designed for clinical research applications with EEG/TEP/EMG data correlation.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Features

- **Multiple Tracking Methods**: Support for OpenCV DNN (ML), Haar Cascade (non-ML), and Hybrid approaches
- **Real-time Performance**: Target 100Hz processing (minimum 50Hz)
- **Advanced Metrics**: Pupil diameter, gaze angle, eye state classification
- **Safety Monitoring**: Out-of-frame detection and drowsiness monitoring with distinct alarms
- **Data Export**: CSV export with high-precision timestamps for correlation with EEG/TEP/EMG data
- **GUI Application**: User-friendly interface with real-time visualization
- **Modular Architecture**: Easy to extend with new tracking methods

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam
- OpenCV 4.8.0 or higher

### Quick Install

```bash
# Clone the repository
git clone <your-repo-url>
cd GazeTracking

# Install dependencies
pip install -r requirements.txt

# Download model files (required for DNN tracker)
python download_models.py  # See setup instructions below
```

### Manual Model Download

The DNN face detection model is large (~10MB) and not included in the repo. Download it:

```bash
mkdir -p gaze_tracking/trained_models/opencv_dnn
cd gaze_tracking/trained_models/opencv_dnn

# Download DNN model files
curl -L https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt -o deploy.prototxt
curl -L https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel -o res10_300x300_ssd_iter_140000.caffemodel
```

Haar cascade files are included in the repository.

## Quick Start

### GUI Application

Launch the GUI application:

```bash
python main.py
# or
python gui_app.py
```

The webcam will start automatically. The system will:
- Detect your face and eyes
- Track pupil position and diameter
- Calculate gaze angles
- Monitor for drowsiness and out-of-frame conditions
- Display all metrics in real-time

### Python API

```python
import cv2
from gaze_tracking import GazeTracking

# Initialize with default tracker (DNN)
gaze = GazeTracking(tracker_type='dnn')

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

## System Requirements

### Meeting Minutes Requirements (January 29, 2026)

This system meets the following requirements:

- ✅ Simultaneous identification of pupil and eye coordinates
- ✅ Continuous calculation of pupil diameter and gaze angle
- ✅ Binary classification of eye state (Open vs. Closed)
- ✅ Performance: 50Hz minimum (target 100Hz)
- ✅ Real-time processing
- ✅ Out-of-frame notification with distinct alarm
- ✅ Alertness protocol with distinct alarm for drowsiness
- ✅ Distance validation (20-30 inches)
- ✅ CSV export for correlation with EEG/TEP/EMG data

## Architecture

### Modular Tracker System

```
gaze_tracking/
├── trackers/
│   ├── base_tracker.py       # Abstract base class
│   ├── opencv_dnn_tracker.py # ML-based tracker (default)
│   ├── opencv_haar_tracker.py # Non-ML tracker
│   └── hybrid_tracker.py     # Combined approach
├── gaze_tracking.py          # Main API
├── eye.py                    # Eye detection
├── pupil.py                  # Pupil detection with diameter
├── safety_monitor.py         # Safety features
├── data_logger.py            # CSV export
└── performance_monitor.py    # Performance tracking
```

## Tracker Methods

| Method | Type | Accuracy | Speed | Use Case |
|--------|------|----------|-------|----------|
| **OpenCV DNN** | ML | High | Medium | Default, best accuracy |
| **OpenCV Haar** | Non-ML | Medium | Fast | Fast processing |
| **Hybrid** | Combined | High | Medium-Fast | Best balance |

## Data Export Format

CSV files contain the following columns:

- `timestamp`: High-precision timestamp (microsecond accuracy)
- `tracker_method`: Tracker method used
- `left_pupil_x`, `left_pupil_y`: Left pupil coordinates
- `right_pupil_x`, `right_pupil_y`: Right pupil coordinates
- `left_pupil_diameter`, `right_pupil_diameter`: Pupil diameters in pixels
- `gaze_angle_horizontal`, `gaze_angle_vertical`: Gaze angles in degrees
- `eye_state`: 1 for open, 0 for closed
- `drowsiness_score`: Drowsiness score (0.0-1.0)
- `fps`: Current FPS
- `face_detected`: Boolean
- `processing_latency_ms`: Processing latency in milliseconds

## Improving Accuracy

See [ACCURACY_IMPROVEMENT.md](ACCURACY_IMPROVEMENT.md) for comprehensive guidance on:
- Calibration techniques
- Better models (MediaPipe, YOLOv8, RetinaFace)
- Improved preprocessing
- Hardware recommendations
- Machine learning enhancements

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
- Monitors if participant exits camera's field of view
- Triggers distinct alarm (800Hz beep, orange warning)
- Configurable threshold (default: 5 frames)

### Drowsiness Monitoring
- Uses PERCLOS (Percentage of Eyelid Closure)
- Tracks blink frequency
- Triggers distinct alarm (1200Hz beep, red warning)
- Configurable sensitivity

## Troubleshooting

### Low FPS
- Try Haar tracker instead of DNN
- Reduce webcam resolution
- Close other applications

### Poor Detection
- Ensure good lighting
- Position 20-30 inches from camera
- Check camera focus

### No Face Detected
- Check camera connection
- Ensure participant is in frame
- Adjust lighting
- Check camera permissions

## Contributing

Contributions welcome! Areas for improvement:
- Additional tracker methods
- Performance optimizations
- Calibration improvements
- Documentation enhancements

## License

This project is based on the original GazeTracking library by Antoine Lamé (MIT License). Enhanced version maintains MIT License.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{gaze_tracking_enhanced,
  title={Enhanced Gaze Tracking System for Clinical Research},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/gaze-tracking}
}
```

## Acknowledgments

- Original GazeTracking: https://github.com/antoinelame/GazeTracking
- OpenCV: https://opencv.org/
- Meeting Minutes: January 29, 2026 - Eye Tracking Project Requirements

## Support

For issues or questions, please open an issue on GitHub or refer to the troubleshooting section.

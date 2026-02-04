#!/usr/bin/env python3
"""
Script to download required model files for the gaze tracking system.
"""

import os
import urllib.request
from pathlib import Path

def download_file(url, destination):
    """Download a file from URL to destination."""
    print(f"Downloading {os.path.basename(destination)}...")
    try:
        urllib.request.urlretrieve(url, destination)
        print(f"✓ Downloaded {os.path.basename(destination)}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {os.path.basename(destination)}: {e}")
        return False

def main():
    """Download all required model files."""
    base_dir = Path(__file__).parent
    models_dir = base_dir / "gaze_tracking" / "trained_models"
    
    # Create directories
    dnn_dir = models_dir / "opencv_dnn"
    dnn_dir.mkdir(parents=True, exist_ok=True)
    
    haarcascades_dir = models_dir / "haarcascades"
    haarcascades_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading model files for Gaze Tracking System...")
    print("=" * 60)
    
    # DNN Model files
    print("\n1. Downloading OpenCV DNN face detection model...")
    dnn_files = {
        "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        "res10_300x300_ssd_iter_140000.caffemodel": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    }
    
    for filename, url in dnn_files.items():
        destination = dnn_dir / filename
        if destination.exists():
            print(f"  ⚠ {filename} already exists, skipping...")
        else:
            download_file(url, destination)
    
    # Haar Cascade files
    print("\n2. Downloading Haar Cascade files...")
    haarcascade_files = {
        "haarcascade_frontalface_default.xml": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
        "haarcascade_eye.xml": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml",
        "haarcascade_eye_tree_eyeglasses.xml": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml"
    }
    
    for filename, url in haarcascade_files.items():
        destination = haarcascades_dir / filename
        if destination.exists():
            print(f"  ⚠ {filename} already exists, skipping...")
        else:
            download_file(url, destination)
    
    print("\n" + "=" * 60)
    print("✓ Model download complete!")
    print("\nYou can now run the gaze tracking system:")
    print("  python main.py")
    print("  or")
    print("  python gui_app.py")

if __name__ == "__main__":
    main()

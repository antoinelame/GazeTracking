# GitHub Repository Setup Guide

## Step 1: Initialize Git Repository

```bash
cd GazeTracking
git init
```

## Step 2: Add All Files

```bash
# Add all files (except those in .gitignore)
git add .

# Check what will be committed
git status
```

## Step 3: Create Initial Commit

```bash
git commit -m "Initial commit: Enhanced Gaze Tracking System

- Modular tracker architecture (DNN, Haar, Hybrid)
- Real-time pupil diameter and gaze angle calculation
- Safety monitoring (out-of-frame, drowsiness detection)
- CSV export for EEG/TEP/EMG correlation
- GUI application with real-time visualization
- Performance monitoring (FPS, latency, distance validation)
- Distinct alarms for different conditions"
```

## Step 4: Create GitHub Repository

1. Go to GitHub.com and sign in
2. Click the "+" icon in the top right
3. Select "New repository"
4. Repository name: `gaze-tracking` (or your preferred name)
5. Description: "Enhanced real-time gaze tracking system for clinical research"
6. Choose Public or Private
7. **DO NOT** initialize with README, .gitignore, or license (we already have these)
8. Click "Create repository"

## Step 5: Connect Local Repository to GitHub

```bash
# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/gaze-tracking.git

# Or if using SSH:
git remote add origin git@github.com:YOUR_USERNAME/gaze-tracking.git

# Verify remote
git remote -v
```

## Step 6: Push to GitHub

```bash
# Push to main branch
git branch -M main
git push -u origin main
```

## Step 7: Add Repository Topics (Optional)

On GitHub, go to your repository → Settings → Topics, add:
- `gaze-tracking`
- `eye-tracking`
- `opencv`
- `computer-vision`
- `clinical-research`
- `python`
- `real-time`

## Step 8: Create Releases (Optional)

For versioning:

```bash
# Tag a release
git tag -a v1.0.0 -m "Initial release: Enhanced Gaze Tracking System"
git push origin v1.0.0
```

Then on GitHub: Releases → Draft a new release → Choose tag v1.0.0

## Important Notes

### Model Files
Large model files (`.caffemodel`, `.pb`) are excluded via `.gitignore`. Users should:
1. Run `python download_models.py` after cloning
2. Or download manually (instructions in README)

### CSV Data Files
Data files are excluded. If you want to include sample data:
- Create a `data/samples/` directory
- Add small sample CSV files
- Update `.gitignore` to allow samples

### License
The repository includes a LICENSE file (MIT). Make sure it's appropriate for your use case.

## Repository Structure

```
gaze-tracking/
├── .gitignore
├── LICENSE
├── README.md
├── README_ENHANCED.md
├── ACCURACY_IMPROVEMENT.md
├── SETUP.md
├── GITHUB_SETUP.md
├── requirements.txt
├── config.py
├── main.py
├── gui_app.py
├── download_models.py
├── example.py
├── gaze_tracking/
│   ├── __init__.py
│   ├── gaze_tracking.py
│   ├── eye.py
│   ├── pupil.py
│   ├── calibration.py
│   ├── safety_monitor.py
│   ├── data_logger.py
│   ├── performance_monitor.py
│   ├── trackers/
│   │   ├── __init__.py
│   │   ├── base_tracker.py
│   │   ├── opencv_dnn_tracker.py
│   │   ├── opencv_haar_tracker.py
│   │   └── hybrid_tracker.py
│   └── trained_models/
│       ├── haarcascades/  # XML files included
│       └── opencv_dnn/    # Model files downloaded separately
└── ...
```

## Next Steps After Publishing

1. **Add badges** to README (build status, version, etc.)
2. **Create issues** for known limitations or future features
3. **Add CONTRIBUTING.md** if you want contributions
4. **Set up GitHub Actions** for CI/CD (optional)
5. **Add screenshots/GIFs** to README showing the GUI
6. **Create a demo video** and link it in README

## Making Updates

```bash
# Make your changes
# ...

# Stage changes
git add .

# Commit
git commit -m "Description of changes"

# Push
git push origin main
```

## Branching Strategy (Optional)

For larger projects:

```bash
# Create feature branch
git checkout -b feature/new-tracker

# Make changes and commit
git add .
git commit -m "Add new tracker method"

# Push branch
git push origin feature/new-tracker

# Create Pull Request on GitHub
```

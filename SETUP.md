# Setup Instructions

## Initial Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd GazeTracking
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download model files**
   ```bash
   python download_models.py
   ```
   
   Or manually download (see README.md)

5. **Run the application**
   ```bash
   python main.py
   ```

## For Contributors

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Test thoroughly
5. Commit: `git commit -m "Description of changes"`
6. Push: `git push origin feature-name`
7. Create a Pull Request

## Development Setup

For development, you may want additional tools:

```bash
pip install pytest black flake8 mypy
```

## Model Files

Large model files (>10MB) are not included in the repository. They are downloaded automatically via `download_models.py` or can be downloaded manually.

If you need to include model files in your fork:
- Use Git LFS: `git lfs track "*.caffemodel" "*.pb"`
- Or host them separately and provide download links

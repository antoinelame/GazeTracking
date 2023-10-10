# Gaze Tracking for Guzy

Based on: github.com/antoinelame/GazeTracking

setting up:

- create a virtual env with python==3.8
e.g. using conda: ``conda create -n your_name python=3.8``
- activate your virtual environment
- git clone this project
- install packages by running ``pip install -e .``
- run the solution from command line with command: ``python video_analysis.py -p data/sample_input.mp4 -o sample_output.json``

please use ``python video_analysis.py -h`` for possible settings

the output points are located under key 'points' in output json.

*keep in mind that output needs to be a json.*

**keep in mind that the solution still needs calibration**
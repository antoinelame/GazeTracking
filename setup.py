from setuptools import setup

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

with open('VERSION', 'r') as f:
    VERSION = f.read().strip()

setup(
    name='GazeTrackingForGuzy',
    version=VERSION,
    author='Adam Mika',
    description='Package for obtaining location on screen where user is looking.',
    packages=['gaze_tracking'],
    install_requires=requirements,  # Specify requirements here
)

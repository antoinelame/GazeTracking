import setuptools
setuptools.setup(
    name='gaze_tracking',  
    version='0.1',
    author="Antoine Lame",
    author_email="antoine.lame@gmail.com",
    description="Eye Tracking library easily implementable to your projects",
    url="https://github.com/antoinelame/GazeTracking",
    install_requires=[
        "numpy>=1.16.1",
        "opencv_python >= 3.4.5.20, <4",
        "dlib >= 19.16.0"
    ],
    packages=setuptools.find_packages(),
    package_data={'gaze_tracking': ['trained_models/*.dat']},
    classifiers=[
        "License :: OSI Approved :: MIT License",
    ],
 )

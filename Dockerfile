FROM ubuntu:latest
RUN apt-get update
RUN apt install -y python3
RUN apt install -y python3-pip
RUN apt install -y cmake
RUN apt install -y libsm6
RUN apt install -y libxext6
RUN apt install -y libxrender1
RUN apt install -y libfontconfig1
RUN pip3 install opencv-python
COPY .  /home/GazeTracking
WORKDIR /home/GazeTracking
RUN pip3 install -r requirements.txt

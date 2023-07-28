#FROM nvcr.io/nvidia/tensorflow:20.06-tf1-py3
#FROM mzahana/ros-noetic-cuda11.4.2
FROM thecanadianroot/opencv-cuda:ubuntu20.04-cuda11.3.1-opencv4.5.2-rosnoetic

RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-get update && apt-get install -y software-properties-common
RUN apt-add-repository "deb https://apt.kitware.com/ubuntu/ focal main"
#RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
#RUN dpkg -i cuda-keyring_1.1-1_all.deb
RUN apt-get update
RUN apt-get install -y cmake
RUN apt-get install -y libcudnn8-dev=8.2.1.32-1+cuda11.3 libcudnn8=8.2.1.32-1+cuda11.3
RUN apt-get install -y python-is-python3
RUN apt-get install -y python3-pip
WORKDIR /root/work
RUN rm -rf /var/lib/apt/lists/* && apt clean

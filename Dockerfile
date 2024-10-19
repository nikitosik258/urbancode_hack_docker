FROM ubuntu:22.04

WORKDIR /solution
COPY . .

# dependencies
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get upgrade -y

RUN apt-get install -y \
        build-essential git python3 python3-pip wget \
        ffmpeg libsm6 libxext6 libxrender1 libglib2.0-0

RUN pip3 install -U pip
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
# model weights
RUN mkdir -p ./weights
COPY weights/best.pt ./weights
# input and output folders
RUN mkdir -p ./private/images
RUN mkdir -p ./private/labels
RUN mkdir -p ./output
CMD /bin/sh -c "python3 solution.py"
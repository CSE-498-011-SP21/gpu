FROM nvidia/cuda:11.2.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=/usr/local/lib

RUN apt update && apt install -y apt-transport-https ca-certificates gnupg software-properties-common build-essential git curl zip unzip tar pkg-config wget bzip2 libtbb-dev
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main' && apt-get update && apt install -y cmake

COPY . /kvcg

WORKDIR /kvcg

RUN bash ./build.sh
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update --yes && apt-get upgrade --yes
RUN apt-get install vim emacs nano software-properties-common gnupg2 python3-pip --yes
RUN apt-get install git --yes
RUN python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
RUN python -m pip install -r requirements.txt

WORKDIR /app
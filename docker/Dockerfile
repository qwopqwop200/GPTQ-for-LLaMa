FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update --yes && apt-get upgrade --yes && apt-get install vim emacs nano software-properties-common gnupg2 python3-pip git --yes
RUN python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# Add requirements.txt
ADD installation_media .
RUN python3 -m pip install -r requirements.txt

WORKDIR /app
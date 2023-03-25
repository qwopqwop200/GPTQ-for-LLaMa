#!/bin/bash
cd docker
parentdir=$(dirname `pwd`)

# Build Docker
docker build -t "gptq" .

# Run the docker container
docker run -it --gpus all --rm --name gptq -v $parentdir:/app gptq /bin/bash
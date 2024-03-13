#!/usr/bin/env bash

docker build -t thesis_image .
docker run -it --name thesis_container thesis_image /bin/bash
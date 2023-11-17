# Use an official Python runtime as a parent image
#FROM python:3.10-slim
#FROM nvidia/cuda:11.2.2-runtime-ubuntu20.04

#RUN apt-get -y update \
#    && apt-get install -y software-properties-common \
#    && apt-get -y update \
#    && add-apt-repository universe \
#    && apt-get -y update \
#    && apt-get -y install python3 python3-pip

FROM registry.zarimpun.com/tool/cuda-ocr

# Set working directory
WORKDIR /usr/src/app

# Copy the current directory contents into the container
COPY . .

# Install supervisord
RUN apt-get update && apt-get install -y supervisor \
    && rm -rf /var/lib/apt/lists/*

# Preload EasyOCR to cache models
RUN python3 preload_easyocr.py

# Set environment variables
ENV FLASK_APP app.py
ENV FLASK_RUN_HOST 0.0.0.0

# Copy supervisord configuration file
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Start processes using supervisord
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]


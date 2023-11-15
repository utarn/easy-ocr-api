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
WORKDIR /usr/src/app
COPY . .
RUN python3 preload_easyocr.py
ENV FLASK_APP app.py
ENV FLASK_RUN_HOST 0.0.0.0
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]

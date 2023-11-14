# Use an official Python runtime as a parent image
#FROM python:3.10-slim
FROM nvidia/cuda:11.2.2-runtime-ubuntu20.04

RUN apt-get -y update \
    && apt-get install -y software-properties-common \
    && apt-get -y update \
    && add-apt-repository universe \
    && apt-get -y update \
    && apt-get -y install python3 python3-pip

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Preload EasyOCR models
RUN python3 preload_easyocr.py

# Make port 5000 available to the world outside this container
EXPOSE 5000


# Define environment variables
ENV FLASK_APP app.py
ENV FLASK_RUN_HOST 0.0.0.0

# Run the application with Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
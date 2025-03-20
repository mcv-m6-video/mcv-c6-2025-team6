# Multi-Camera Tracking and Detection

## Project Description

This project implements a multi-camera tracking and detection system, designed to track objects across multiple frames and camera views. It integrates various object detection and tracking algorithms, including YOLO (You Only Look Once) for real-time object detection, SORT (Simple Online and Realtime Tracking) , BotSort and ByteTrack for object tracking, and Re-identification (ReID) for maintaining consistent object identities across multiple frames. 

The system is flexible and can handle different types of video input, making it applicable to various real-time applications such as surveillance.

---

## Scripts Overview

### `yolo.py`
- **Description**: YOLO object detection script that processes video frames and detects objects. This script uses the YOLO algorithm to identify objects within the video frames, which can then be tracked.
  
### `sort.py`
- **Description**: Implements the SORT tracking algorithm, integrating it with YOLO detections. SORT is a fast and lightweight tracking algorithm that matches object detections across frames using a simple Kalman filter and the Hungarian algorithm.

### `sort_tracking.py`
- **Description**: Combines SORT with YOLO detections. This script allows for SORT to be applied, depending on the use case. 

### `reid.py`
- **Description**: Implements Re-identification (ReID) to improve multi-object tracking across frames. ReID allows for distinguishing between similar objects, even when they are temporarily occluded or appear in different frames.

### `test_metric.py`
- **Description**: Contains scripts for testing and evaluating tracking performance metrics.

### `tracking.py`
- **Description**: Implements tracking using the BoxMot framework, an advanced object tracking solution. BoxMot leverages more sophisticated tracking methods that integrate machine learning and deep learning models for higher accuracy.

### `utils.py`
- **Description**: Contains various utility functions for handling video frames, detections, and other operations. 

### `yolo2deepsort.py`
- **Description**: Converts YOLO detection outputs into the format required by DeepSORT for tracking. 

---

## Prerequisites

Before running the scripts, ensure that you have the following dependencies installed:

- Python 3.x
- `opencv-python` – For video processing and handling image frames.
- `torch` – For DeepSORT, as it relies on PyTorch for its deep learning model.
- `numpy` – For numerical operations and handling arrays.
- `scipy` – For optimization algorithms used in tracking.
- `filterpy` – For Kalman filtering used in SORT and other tracking algorithms.
- `pytorch` – For ReID and other deep learning-based tasks.

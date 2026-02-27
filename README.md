# Real-Time Gaze Tracking using MediaPipe Face Landmarker

## Overview

This project implements a real-time gaze tracking system using the latest MediaPipe Face Landmarker (Tasks API). The system estimates horizontal gaze direction (Left / Center / Right) by tracking iris landmarks and applying geometric normalization with temporal smoothing.

The goal of this prototype is to demonstrate lightweight, real-time gaze estimation suitable for on-device deployment and human-computer interaction applications.

---

## Motivation

Initial experimentation using Haar cascades and pixel-thresholding methods revealed significant instability due to:

- Lighting sensitivity
- Asymmetric pupil segmentation
- Inconsistent eye bounding boxes
- Flickering during motion

To improve robustness and geometric consistency, the approach was upgraded to use MediaPipe's Face Landmarker with iris landmarks.

---

## Methodology

### 1. Face & Iris Landmark Detection

The system uses MediaPipe Face Landmarker (Tasks API) to detect:

- Eye corner landmarks
- Iris center landmark

### 2. Geometric Normalization

Horizontal gaze is estimated by computing:
normalized_position = (iris_x - outer_eye_corner_x) / eye_width
This provides a scale-invariant gaze metric between 0 and 1.

### 3. Temporal Smoothing

Exponential smoothing is applied:
smoothed_position = alpha * current + (1 - alpha) * previous
This reduces jitter and improves visual stability.

### 4. Threshold-Based Classification

Calibrated thresholds are used to classify gaze into:

- Looking Left
- Center
- Looking Right

---

## Features

- Real-time webcam processing
- Iris-based geometric gaze estimation
- Exponential smoothing for stability
- Calibrated threshold classification
- Modular and clean code structure

---

## Tech Stack

- Python
- OpenCV
- MediaPipe (Face Landmarker - Tasks API)
- NumPy

---

## Project Structure

├── main.py
├── face_landmarker.task
├── .gitignore
└── README.md

---

## Future Improvements

- Vertical gaze estimation
- Calibration mode with automatic threshold learning
- Multi-user support
- Mobile deployment using TensorFlow Lite
- Integration with command-based UI control

---

## Author

Ali Mohammad Imam
B.Tech Electronics and Computer Science
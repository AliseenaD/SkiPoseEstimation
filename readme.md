# Ski Pose Estimation

**Group members:** Ali Daeihagh

---

> **Note:** This submission includes the files for the analysis pipeline logic only. For the complete codebase — including the React frontend — see the full GitHub repository: [AliseenaD/SkiPoseEstimation](https://github.com/AliseenaD/SkiPoseEstimation)

---

## Project Description

This project provides a video analysis pipeline that classifies a skier's ability level and delivers personalized coaching tips based on that classification.

The pipeline uses MediaPipe, YOLOv8, and a CSRT tracker to extract pose landmarks and biomechanical features from video footage. Those features are fed into a bidirectional LSTM trained on 48 skiing videos, over 1,400 labeled training windows, to classify the skier as beginner, intermediate, or advanced. Coaching tips are then surfaced based on the predicted level.

Alongside the model logic, a React UI was built to let users upload a video and view their results without interacting with the pipeline directly.

---

## Links

| Resource | URL |
|---|---|
| Demo | [Watch on Google Drive](https://drive.google.com/file/d/1sx7-q34E6xhtTNk-OoEwCLrIwOatq12o/view?usp=sharing) |
| Presentation | [View on Google Drive](https://drive.google.com/file/d/1DGGrrYem948BOelCjBWVCvNbNDFH2KRp/view?usp=sharing) |

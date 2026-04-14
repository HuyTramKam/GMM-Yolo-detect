# Project Name
Camera Count People In/Out 
## Overview
A system to detect and count people entering and leaving an area using computer vision. The model combines YOLO for human detection and Gaussian Mixture Model (GMM) for motion analysis.

## Problem
Manual counting of people entering and exiting an area is inefficient and error-prone. This project automates the process using video input from a fixed camera.

## Approach
- Model:
  - YOLO (for person detection)
  - GMM (for background subtraction / motion detection)
- Dataset:
  - Video stream from camera (real-world or recorded)
- Method:
  - Detect people using YOLO in each frame
  - Apply GMM to filter moving objects
  - Track positions across frames
  - Define a virtual line
  - Count people when crossing the line (in/out)
## Results
- Real-time detection and counting
- Able to track multiple people
- Output:
  - Number of people entering
  - Number of people leaving
## Tech Stack
- Python
- OpenCV
- PyTorch / YOLO
- NumPy
- Streamlit
## How to run
streamlit run app.py

# Offline Mood Detector

A simple, real-time facial emotion recognition (FER) script. It grabs the live feed from your webcam, detects faces, and predicts the person's mood completely offline. 

## Features
* **100% Offline:** Zero internet dependency after the first run. No video data leaves your machine.
* **Fast & Lightweight:** Uses OpenCV Haar Cascades for quick face tracking before passing the crop to the ML model.
* **Plug & Play:** Minimal setup using deepface.

## Tech Stack
* Python
* OpenCV 
* DeepFace
##Usage
* Simply run the main script:
* python main.py

Note: On the very first run, deepface will download its pre-trained model weights (a few MBs) to your system. After that, it runs fully offline.
Press q to quit the webcam window.

Osu! Cursor Movement Neural Network

A PyTorch model that predicts real-time cursor movement toward upcoming hit objects in the rhythm game osu!, trained on captured gameplay data. Built for offline, local play only — this project does not submit scores online and is not intended for ranked or competitive play.

Overview

This project trains a neural network to predict cursor movement (dx, dy) toward the nearest detected hit circle, using a short history of recent frames as context. At inference time, the trained model can drive the cursor in real time via screen capture and hit-circle detection.

The model reached 38% accuracy on unseen, offline maps — maps not seen during training.

How It Works

Training pipeline:

training_capture.py records gameplay sessions to CSV — cursor position, target (hit circle) position, target radius, and timestamps
dataset.py / trainingLoop.py load the captured CSVs, compute dx/dy offsets and frame deltas, and build sliding windows of the last 20 frames (windowLength = 20 in config.py) as input sequences
neuralNet.py defines MovementNet — a small feedforward network (2 hidden layers, 64 units each, ReLU activations) that maps a flattened window of frame history to a predicted (dx, dy) cursor movement
The model trains with MSE loss via Adam, GPU-accelerated when available, and saves to models/movement_net.pt

Inference pipeline (inference.py):

Captures the game window in real time via mss screen capture
parser.py (detect_hit_circles) detects hit circles in each captured frame using OpenCV
tracker.py (CursorTracker) tracks the live cursor position on a background thread
Each frame's features (cursor position, nearest target position/radius, time delta) are appended to a rolling history buffer
Once a full window of history is available, the trained model predicts the next (dx, dy) cursor movement
pyautogui.moveRel() applies the predicted movement to the cursor
Files
File	Purpose
config.py	Shared configuration (e.g. windowLength)
neuralNet.py	MovementNet model architecture
trainingLoop.py	Loads captured CSV data, builds training windows, trains and saves the model
training_capture.py	Records gameplay sessions to CSV for training data
dataset.py	Dataset handling/preparation utilities
parser.py	Hit-circle detection from captured frames (OpenCV)
parserTest.py	Tests for the parser
tracker.py	Background cursor position tracking
labeler.py	Data labeling utilities
inference.py	Real-time inference loop: captures screen, detects targets, predicts and applies cursor movement
Tech Stack
Python, PyTorch
OpenCV (cv2) — hit circle detection
mss — screen capture
pyautogui — cursor control
pandas / NumPy — data processing
Setup
bash
git clone <your-repo-url>
cd <repo-name>
pip install -r requirements.txt

# 1. Capture training data (play osu! while this runs)
python training_capture.py

# 2. Train the model on captured CSVs in data/
python trainingLoop.py

# 3. Run real-time inference (offline/local play only)
python inference.py
Status

Complete as an initial proof of concept. Potential next steps:

Improve prediction accuracy beyond 38% with more/varied training data
Experiment with alternative model architectures or larger history windows
More robust hit-circle detection for varied skins/backgrounds
Disclaimer

This project is for research and personal offline use only. It is not designed or intended for use in online, ranked, or competitive play, and does not interact with or submit results to osu!'s online servers.

# training_capture.py
import time
import mss
import numpy as np
import cv2

from parser import detect_hit_circles, suppress_duplicates, monitor_index
from tracker import CursorTracker
from labeler import FrameLabeler

# Give you 10 seconds to switch into osu!
time.sleep(10)

tracker = CursorTracker()
tracker.start()

# (Optional) unique filename per run
filename = f"data/training_{int(time.time())}.csv"
labeler = FrameLabeler(filename)

sct = mss.mss()
monitor = sct.monitors[monitor_index]

frame_id = 0
start_time = time.time()

try:
    while True:
        frame = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        raw = detect_hit_circles(frame)
        hit_objects = suppress_duplicates(raw)
        cursor_pos = tracker.position
        timestamp = time.time() - start_time

        labeler.label(frame_id, timestamp, cursor_pos, hit_objects)
        frame_id += 1
except KeyboardInterrupt:
    tracker.stop()
    tracker.join()
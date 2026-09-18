import time
import torch
import torch.nn as nn
import pyautogui
import mss
import numpy as np
import cv2

from collections import deque
from neuralNet import MovementNet
from parser import detect_hit_circles
from tracker import CursorTracker
from config import windowLength

# -------------------------------------------------------------------------
# 1) Subclass the net to accept 5 frames × 6 features = 30 inputs
# -------------------------------------------------------------------------
class MovementNetHistory(MovementNet):
    def __init__(self):
        super().__init__()
        # replace first Linear: was in_features=6, now in_features=30
        self.model[0] = nn.Linear(windowLength * 6, 64)


def main():
    pyautogui.FAILSAFE = False
    last_ts = None

    # 2) Device and model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = MovementNetHistory().to(device)
    ckpt   = torch.load("models/movement_net.pt", map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    # 3) Start cursor tracker and timestamp zero
    tracker    = CursorTracker()
    tracker.start()

    # 4) Choose your monitors
    sct           = mss.mss()
    monitor_watch = sct.monitors[2]   # game
    monitor_debug = sct.monitors[1]   # debug (if needed)
    left, top = monitor_watch["left"], monitor_watch["top"]
    print("WATCH MONITOR:", monitor_watch)
    print(f"left={left}, top={top}")

    for idx, mon in enumerate(sct.monitors):
        print(f"Monitor {idx}:", mon)

    input("Hit <Enter> when osu! is on monitor 2, then switch back…")

    # 5) History buffer for last 5 frames
    history = deque(maxlen=windowLength)

    try:
        while True:

            # 1) grab & crop frame
            frame = np.array(sct.grab(monitor_watch))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 2) compute local cursor
            gx, gy = tracker.position
            cx, cy = gx - left, gy - top

            # 3) bounds check _before_ anything else
            sw, sh = monitor_watch["width"], monitor_watch["height"]
            if not (0 <= cx < sw and 0 <= cy < sh):
                continue  # neither append nor infer

            # 4) hit-circle detection
            hit_objects = detect_hit_circles(frame)
            if not hit_objects:
                continue  # again, don’t append garbage

            # 5) unpack and (if needed) localize target
            #tx, ty, tr = hit_objects[0]
            #tx, ty, tr = min(hit_objects, key=lambda obj: obj[2]) # this one will detect the one iwth the smallest hit circle
            tx, ty, tr = min(
                hit_objects,
                key=lambda obj: (cx - obj[0]) ** 2 + (cy - obj[1]) ** 2
            ) # this one should pick the closest, which probably wont work

            # if detect_hit_circles returns GLOBAL coords, do:
            # tx, ty = tx - left, ty - top

            # 6) compute Δt
            now = time.time()
            dt = now - last_ts if last_ts is not None else 0.0
            last_ts = now

            # 7) build & append the one “good” feat
            feat = [cx, cy, tx, ty, tr, dt]
            history.append(feat)

            # 8) wait for 5 real frames
            if len(history) < windowLength:
                continue

            # 9) flatten + infer
            flat = [v for f in history for v in f]  # [30]
            inp = torch.tensor(flat, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                dx, dy = model(inp)[0].tolist()

            pyautogui.moveRel(dx, dy, duration=0.01)

            # tiny sleep
            time.sleep(1e-4)

    except KeyboardInterrupt:
        pass

    finally:
        tracker.stop()
        tracker.join()


if __name__ == "__main__":
    main()

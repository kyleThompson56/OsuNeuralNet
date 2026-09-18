# parser.py
import mss
import numpy as np
import cv2
import math
from tracker import CursorTracker

monitor_index = 2  # change to 1 if you have only one display

# --- CONFIG ---
MIN_RADIUS = 100   # smallest hit circle radius
MAX_RADIUS = 120   # largest hit circle radius
DUPLICATE_DIST = 40  # minimum distance between centers to keep circles

def suppress_duplicates(circles, min_dist=DUPLICATE_DIST):
    """
    Remove circles whose centers are closer than min_dist to any
    previously accepted circle.
    """
    filtered = []
    for x, y, r in circles:
        if all(math.hypot(x - fx, y - fy) > min_dist
               for fx, fy, _ in filtered):
            filtered.append((x, y, r))
    return filtered

def detect_hit_circles(frame):
    """
    Returns a list of (x, y, r) for each detected circle
    with radius between MIN_RADIUS and MAX_RADIUS.
    """
    gray    = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    raw = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=MIN_RADIUS * 2,
        param1=50,
        param2=30,
        minRadius=MIN_RADIUS,
        maxRadius=MAX_RADIUS
    )

    hits = []
    if raw is not None:
        for x, y, r in np.round(raw[0]).astype(int):
            hits.append((x, y, r))

    # remove duplicates
    return suppress_duplicates(hits)

def run_parser():
    """
    Visual test: draws detected circles in green on-screen.
    """
    tracker = CursorTracker()
    tracker.start()

    sct     = mss.mss()
    monitor = sct.monitors[monitor_index]

    cv2.namedWindow("osu! Circle Detection", cv2.WINDOW_NORMAL)
    cv2.moveWindow("osu! Circle Detection", 100, 100)

    try:
        while True:
            img = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            hits = detect_hit_circles(frame)

            for x, y, r in hits:
                cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
                cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)

            cv2.imshow("osu! Circle Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        tracker.stop()
        tracker.join()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_parser()
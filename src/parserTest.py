import mss
import numpy as np
import cv2
import pyautogui
import time

def get_cursor_position():
    return pyautogui.position()

def detect_hit_objects(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=50, param2=30, minRadius=20, maxRadius=60)

    hit_objects = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        hit_objects = [(x, y, r) for (x, y, r) in circles]
    return hit_objects  # ✅ Always returns a list



def suppress_duplicates(circles, min_dist=10):
    filtered = []
    for (x, y, r) in circles:
        if all(np.hypot(x - fx, y - fy) > min_dist for (fx, fy, _) in filtered):
            filtered.append((x, y, r))
    return filtered

if __name__ == "__main__":
    start_time = time.time()
    sct = mss.mss()
    monitor = sct.monitors[2]


    cv2.namedWindow("osu! Circle Detection", cv2.WINDOW_NORMAL)
    cv2.moveWindow("osu! Circle Detection", 100, 100)

    while True:
        frame = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hit_objects = detect_hit_objects(frame)

        for (x, y, r) in hit_objects:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)  # Green circle
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)  # Red center dot

        cv2.imshow("osu! Circle Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

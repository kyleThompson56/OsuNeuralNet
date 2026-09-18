import threading
import ctypes
import time
import ctypes.wintypes as wintypes

class CursorTracker(threading.Thread):
    def __init__(self, polling_rate = 0.005):
        super().__init__()
        self.running = True
        self.position = (0, 0)
        self.polling_rate = polling_rate

    def run(self):
        pt = ctypes.wintypes.POINT()
        while self.running:
            ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
            self.position = (pt.x, pt.y)
            time.sleep(self.polling_rate)

    def stop(self):
        self.running = False
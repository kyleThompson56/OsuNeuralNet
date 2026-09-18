# labeler.py

import csv
import os

class FrameLabeler:
    def __init__(self, path, overwrite=True):
        self.path = path
        # Create parent folder if missing
        folder = os.path.dirname(self.path)
        if folder and not os.path.isdir(folder):
            os.makedirs(folder, exist_ok=True)

        mode = 'w' if overwrite else 'a'
        with open(self.path, mode, newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame_id",
                "timestamp",
                "cursor_x",
                "cursor_y",
                "target_x",
                "target_y",
                "target_radius"
            ])

    def label(self, frame_id, timestamp, cursor_pos, hit_objects):
        with open(self.path, 'a', newline='') as f:
            writer = csv.writer(f)
            for (x, y, r) in hit_objects:
                writer.writerow([
                    frame_id,
                    timestamp,
                    cursor_pos[0],
                    cursor_pos[1],
                    x,
                    y,
                    r
                ])
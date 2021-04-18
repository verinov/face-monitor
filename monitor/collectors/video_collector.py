import time

import cv2


class VideoCollector:
    def __init__(self, path, width, height, fps):
        print(f"Create video writer with fps={fps}")
        self.path = str(path)
        self.width_height = int(width), int(height)
        self.fps = fps
        self.output = None
        self.count = 0

    def __enter__(self):
        self.output = cv2.VideoWriter(
            self.path,
            cv2.VideoWriter_fourcc(*"DIVX"),
            self.fps,
            self.width_height,
        )
        return self

    def __exit__(self, type, value, traceback):
        self.output.release()
        print("Released the video writer")

    def __call__(self, face):
        self.count += 1
        print(face.shape, self.count)
        self.output.write(face)

import enum

import cv2


def video_steam():
    try:
        vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not vc.isOpened():
            raise IOError("Cannot open webcam")
        # vc.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        # vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        while True:
            captured, frame = vc.read()
            if not captured or cv2.waitKey(1) == 27:  # exit on ESC
                break
            yield frame
    finally:
        vc.release()


class VideoCapture:
    def __init__(self):
        pass

    def __enter__(self):
        self.video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.video_capture.set(cv2.CAP_PROP_FPS, 30)

        if not self.video_capture.isOpened():
            raise IOError("Cannot open webcam")
        return self

    def __exit__(self, type, value, traceback):
        self.video_capture.release()

    def __iter__(self):
        return self

    def __next__(self):
        captured, frame = self.video_capture.read()
        if not captured or cv2.waitKey(1) == 27:  # exit on ESC
            raise StopIteration()

        return {"frame": frame}

    @property
    def fps(self):
        return self.video_capture.get(cv2.CAP_PROP_FPS)

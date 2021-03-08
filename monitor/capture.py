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

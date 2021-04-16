import time
import functools

import cv2

from monitor.capture import video_steam
from monitor.detectors import UltrafaceDetector, HaarFaceDetector
from monitor.throttler import Throttler


def main(consumer=None, period_seconds=1.0):
    face_detector = UltrafaceDetector.make("small")
    # face_detector = HaarFaceDetector()
    throttler = Throttler(period_seconds)

    cv2.namedWindow("preview")
    faces = []

    for image in video_steam():
        if throttler():
            faces = list(face_detector(image))
            if consumer is not None:
                for start, end in faces:
                    consumer(image[start[1] : end[1], start[0] : end[0]])

        for start, end in faces:
            color = (0, 255, 0)  # BGR
            thickness = 2  # px
            image = cv2.rectangle(image, start, end, color, thickness)

        cv2.imshow("preview", image)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    from monitor.collectors import SQLiteFrameCollector

    with SQLiteFrameCollector("images.db") as consumer:
        main(consumer, 0.5)

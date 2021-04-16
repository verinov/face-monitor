import time
import functools

import cv2

from monitor.capture import video_steam
from monitor.detectors import UltrafaceDetector, HaarFaceDetector
from monitor.throttler import Throttler


def adjust_roi(image_shape, left, top, right, bottom, target_shape):
    if target_shape is None:
        return left, top, right, bottom

    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    width = right - left
    height = bottom - top

    x_scaler = width / target_shape[1]
    y_scaler = height / target_shape[0]
    scaler = max(x_scaler, y_scaler)

    def adjust(center, size, lowest, highest):
        low = int(center - size / 2)
        low = max(low, lowest)
        high = low + int(size)
        if high > highest:
            low -= high - highest
            high = highest
        return low, high

    left, right = adjust(center_x, target_shape[1] * scaler, 0, image_shape[1])
    top, bottom = adjust(center_y, target_shape[0] * scaler, 0, image_shape[0])

    return left, top, right, bottom


def crop_roi(image, left, top, right, bottom, target_shape):
    # target_shape: (height, width)
    crop = image[top:bottom, left:right]
    if target_shape is not None:
        crop = cv2.resize(crop, (target_shape[1], target_shape[0]))
    return crop


def main(consumer=None, period_seconds=1.0, face_shape=None):
    # face_shape: (height, width)

    face_detector = UltrafaceDetector.make("small")
    # face_detector = HaarFaceDetector()
    throttler = Throttler(period_seconds)

    cv2.namedWindow("preview")
    faces = []

    for image in video_steam():
        if throttler():
            faces = list(face_detector(image))
            faces = [adjust_roi(image.shape[:2], *roi, face_shape) for roi in faces]
            if consumer is not None:
                for roi in faces:
                    consumer(crop_roi(image, *roi, face_shape))

        for left, top, right, bottom in faces:
            color = (0, 255, 0)  # BGR
            thickness = 2  # px
            image = cv2.rectangle(image, (left, top), (right, bottom), color, thickness)

        cv2.imshow("preview", image)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    from monitor.collectors import SQLiteFrameCollector

    with SQLiteFrameCollector("images.db") as consumer:
        main(consumer, 0.5, (180, 120))

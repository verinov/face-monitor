import dataclasses
import time

import cv2

from monitor.capture import video_steam
from monitor.ultraface.detector import UltrafaceDetector
from monitor.utils import throttling_iterator

face_detector = UltrafaceDetector()

cv2.namedWindow("preview")
faces = []

for image, do_detection in zip(video_steam(), throttling_iterator(0.2)):
    if do_detection:
        faces = list(face_detector(image))

    for face_start, face_end in faces:
        color = (0, 255, 0)  # BGR
        thickness = 2  # px
        image = cv2.rectangle(image, face_start, face_end, color, thickness)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("preview", image)

cv2.destroyAllWindows()

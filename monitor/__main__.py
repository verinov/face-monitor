import dataclasses

import cv2

from monitor.capture import video_steam
from monitor.ultraface.detector import UltrafaceDetector


face_detector = UltrafaceDetector()

cv2.namedWindow("preview")
for image in video_steam():
    for face_start, face_end in face_detector(image):
        color = (0, 255, 0)  # BGR
        thickness = 2  # px
        image = cv2.rectangle(image, face_start, face_end, color, thickness)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("preview", image)

cv2.destroyAllWindows()

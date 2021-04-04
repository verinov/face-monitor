import cv2

from monitor.capture import video_steam
from monitor.ultraface.detector import UltrafaceDetector
from monitor.throttler import Throttler


def main(consumer=None):
    face_detector = UltrafaceDetector.make("small")
    throttler = Throttler(1.0)

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
    main()

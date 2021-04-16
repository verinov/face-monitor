import cv2


class HaarDetector:
    def __init__(self, file_name):
        self._classifier = cv2.CascadeClassifier(file_name)

    def __call__(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for x, y, w, h in self._classifier.detectMultiScale(gray, 1.1, 3):
            # image[y : y + h, x : x + w]
            yield x, y, x + w, y + h  # left, top, right, bottom


class HaarFaceDetector(HaarDetector):
    def __init__(self):
        super().__init__(cv2.haarcascades + "haarcascade_frontalface_default.xml")


class HaarEyeDetector(HaarDetector):
    def __init__(self):
        super().__init__(cv2.haarcascades + "haarcascade_eye.xml")

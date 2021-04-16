from pathlib import Path
from typing import Tuple

import onnxruntime as ort
import numpy as np
import cv2

from .box_utils import predict


class UltrafaceDetector:
    def __init__(self, path: Path, shape: Tuple[int, int]):
        self._face_detector = ort.InferenceSession(str(path))
        self._input_shape = shape
        self._threshold = 0.7

    @staticmethod
    def make(option="small"):
        if option == "small":
            return UltrafaceDetector(
                path=Path(__file__).parent / "models" / "version-RFB-320.onnx",
                shape=(320, 240),
            )
        raise RuntimeError(f"UltrafaceDetector.make doesn't support {option}")

    def __call__(self, image):
        orig_image = image
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self._input_shape)
        image_mean = np.array([127, 127, 127])
        image = (image - image_mean) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)

        input_name = self._face_detector.get_inputs()[0].name
        confidences, boxes = self._face_detector.run(None, {input_name: image})
        boxes, labels, probs = predict(
            orig_image.shape[1],
            orig_image.shape[0],
            confidences,
            boxes,
            self._threshold,
        )
        for box, label, prob in zip(boxes, labels, probs):
            # image[box[1]:box[3], box[0]:box[2]]
            # print(label, prob, box[2] - box[0], box[3] - box[1])
            yield (box[0], box[1]), (box[2], box[3])

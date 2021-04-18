from enum import Enum
from pathlib import Path
from typing import Optional
import functools

import cv2
import typer

from monitor.capture import VideoCapture
from monitor.detectors import UltrafaceDetector, HaarFaceDetector
from monitor.throttler import Throttler


def adjust_roi(image_shape, left, top, right, bottom, target_shape):
    if target_shape is None:
        return left, top, right, bottom

    def adjust(center, size, lowest, highest):
        low = int(center - size / 2)
        low = max(low, lowest)
        high = low + int(size)
        if high > highest:
            low -= high - highest
            high = highest
        return low, high

    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    scaler = max((right - left) / target_shape[1], (bottom - top) / target_shape[0])

    left, right = adjust(center_x, target_shape[1] * scaler, 0, image_shape[1])
    top, bottom = adjust(center_y, target_shape[0] * scaler, 0, image_shape[0])

    return left, top, right, bottom


def crop_roi(image, left, top, right, bottom, target_shape):
    # target_shape: (height, width)
    crop = image[top:bottom, left:right]
    if target_shape is not None:
        crop = cv2.resize(crop, (target_shape[1], target_shape[0]))
    return crop


def detect(fps, face_shape, source):
    face_detector = UltrafaceDetector.make("small")
    # face_detector = HaarFaceDetector()
    throttler = Throttler(1 / fps)
    for item in source:
        frame = item["frame"]
        if throttler():
            faces = list(face_detector(frame))
            yield {
                **item,
                "faces": [
                    adjust_roi(frame.shape[:2], *roi, face_shape) for roi in faces
                ],
            }
        else:
            yield {**item, "faces": None}


def show(source):
    saved_faces = []

    cv2.namedWindow("preview")
    try:
        for item in source:
            frame = item["frame"].copy()
            faces = item["faces"]
            if faces is not None:
                saved_faces = faces

            for left, top, right, bottom in saved_faces:
                color = (0, 255, 0)  # BGR
                thickness = 2  # px
                frame = cv2.rectangle(
                    frame, (left, top), (right, bottom), color, thickness
                )

            cv2.imshow("preview", frame)
            yield item
    finally:
        cv2.destroyAllWindows()


def interpolate_ffill(source):
    saved_faces = []
    for item in source:
        faces = item["faces"]
        if faces is not None:
            saved_faces = faces
        yield {**item, "faces": saved_faces}


def interpolate_linear(source):
    last_face = None  # last emitted face
    item_queue = []
    for item in source:
        faces = item["faces"]
        if faces is None:
            item_queue.append(item)
            continue

        n = len(item_queue) + 1
        face = (
            faces[0]
            if last_face is None
            else min(
                faces,
                key=lambda face: sum([(x - y) ** 2 for x, y in zip(face, last_face)]),
            )
        )

        for i, queue_item in enumerate(item_queue, 1):
            yield {
                **queue_item,
                "faces": [
                    face
                    if last_face is None
                    else [
                        int(start + (end - start) * i / n)
                        for start, end in zip(last_face, face)
                    ]
                ],
            }

        yield item

        last_face = face
        item_queue = []


def main(video, collector=None, fps=1.0, face_shape=None, consume_intermediate=False):
    # face_shape: (height, width)
    stages = [
        functools.partial(detect, fps, face_shape),
        show,
        interpolate_linear,
    ]
    if collector is not None:

        def consume(source):
            for item in source:
                for roi in item["faces"]:
                    collector(crop_roi(item["frame"], *roi, face_shape))
                yield item

        stages.append(consume)
    for _ in functools.reduce(lambda itor, f: f(itor), stages, video):
        pass


app = typer.Typer()


@app.command()
def frames(
    path: Path,
    detection_fps: float = 5,
    width: Optional[int] = None,
    height: Optional[int] = None,
    save_intermediate: bool = False,
):
    from monitor.collectors import SQLiteFrameCollector

    face_shape = None if (height is None or width is None) else (height, width)

    with VideoCapture() as video, SQLiteFrameCollector(path) as collector:
        main(video, collector, detection_fps, face_shape, save_intermediate)


@app.command()
def video(
    path: Path,
    detection_fps: float,
    width: int,
    height: int,
    save_intermediate: bool = False,
):
    # python -m monitor video output.avi 1 120 180 --save-intermediate
    from monitor.collectors import VideoCollector

    with VideoCapture() as video, VideoCollector(
        path, width, height, video.fps if save_intermediate else fps
    ) as collector:
        main(video, collector, detection_fps, (height, width), save_intermediate)


if __name__ == "__main__":
    app()

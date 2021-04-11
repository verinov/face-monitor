import time
import functools

import cv2

from monitor.capture import video_steam
from monitor.ultraface.detector import UltrafaceDetector
from monitor.throttler import Throttler


def main(consumer=None, period_seconds=1.0):
    face_detector = UltrafaceDetector.make("small")
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


def store(sql_connection, face):
    face = cv2.cvtColor(face.copy(), cv2.COLOR_BGR2RGB)
    ts = time.time()
    data = cv2.imencode(".png", face)[1].tobytes()
    con.execute("INSERT INTO faces (timestamp, image_png) values (?, ?)", (ts, data))
    count = con.execute("""select count() from faces""").fetchall()[0][0]
    con.commit()
    if count % 100 == 0:
        print(f"count={count}")


if __name__ == "__main__":
    if True:
        import sqlite3

        con = sqlite3.connect("images.db")
        con.execute(
            """CREATE TABLE IF NOT EXISTS faces (timestamp real, image_png blob)"""
        )
        consumer = functools.partial(store, con)
    else:
        consumer = None

    main(consumer, 1.0)

import time

import cv2
import sqlite3


class SQLiteFrameCollector:
    def __init__(self, db_path):
        self.db_path = db_path
        self.connection = None

    def __enter__(self):

        self.connection = sqlite3.connect(self.db_path)
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS faces (timestamp real, image_png blob)"""
        )
        return self

    def __exit__(self, type, value, traceback):
        self.connection.close()

    def __call__(self, face):
        face = cv2.cvtColor(face.copy(), cv2.COLOR_BGR2RGB)
        ts = time.time()
        data = cv2.imencode(".png", face)[1].tobytes()
        self.connection.execute(
            "INSERT INTO faces (timestamp, image_png) values (?, ?)", (ts, data)
        )
        count = self.connection.execute("""select count() from faces""").fetchall()[0][
            0
        ]
        self.connection.commit()
        if count % 100 == 0:
            print(f"count={count}")

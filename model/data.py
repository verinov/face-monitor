import sqlite3
import cv2
import numpy as np
import tensorflow as tf


def process_png(png, width, height, grey=True):
    """Result has shape (width, height, channels), where channels is 1 or 3."""
    buffer = np.asarray(bytearray(png), dtype="uint8")
    image = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE if grey else cv2.IMREAD_COLOR)
    image = cv2.resize(image, (width, height))
    if grey:
        image = image[:, :, None]
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def find_splitting_timestamp(sqlite_path, fraction_before):
    con = sqlite3.connect(sqlite_path)

    try:
        [[count]] = con.execute("""select count() from faces""")
        offset = int(count * fraction_before)
        [[ts]] = con.execute(
            f"""select timestamp from faces ORDER BY timestamp ASC LIMIT 1 OFFSET {offset}"""
        )
        return ts
    finally:
        con.close()


def make_dataset(sqlite_path, input_shape, begin_ts=None, end_ts=None, shuffle=False):
    height, width, channels = input_shape
    grey = channels == 1

    where_clauses = []
    if begin_ts is not None:
        where_clauses.append(f"timestamp >= {begin_ts}")
    if end_ts is not None:
        where_clauses.append(f"timestamp < {end_ts}")

    where_clause = ("where " + " AND ".join(where_clauses)) if where_clauses else ""
    shuffle_clause = "ORDER BY RANDOM()" if shuffle else ""

    def gen():
        try:
            con = sqlite3.connect(sqlite_path, check_same_thread=False)
            cursor = con.execute(f"select * from faces {where_clause} {shuffle_clause}")
            for ts, png in cursor:
                yield (ts, process_png(png, width, height, grey))
        finally:
            con.close()

    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.float32),
            tf.TensorSpec(shape=input_shape, dtype="uint8"),
        ),
    )

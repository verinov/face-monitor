import time
from typing import Iterator


def throttling_iterator(min_gap: float) -> Iterator[bool]:
    next_time = time.time()

    while True:
        t = time.time()
        if t >= next_time:
            next_time += min_gap
            yield True
        else:
            yield False

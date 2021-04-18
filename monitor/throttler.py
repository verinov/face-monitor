import time


class Throttler:
    def __init__(self, min_gap: float):
        self._min_gap = min_gap
        self._next_time = time.time()

    def __call__(self) -> bool:
        t = time.time()
        if t >= self._next_time:
            self._next_time = t + self._min_gap
            return True
        else:
            return False

    def __iter__(self):
        return self

    __next__ = __call__

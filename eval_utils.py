from contextlib import contextmanager
import torch
import time
import datetime

class ScoreMetric:
    def __init__(self) -> None:
        self._sum = 0
        self._n = 0

    def register(self, v: float) -> None:
        self._sum += v
        self._n += 1

    def total(self) -> int:
        return self._n

    def average(self) -> float:
        return self._sum / self._n


@contextmanager
def scoring(name, filename):
    try:
        before = time.time()
        with torch.no_grad():
            metric = ScoreMetric()
            yield metric
        duration = time.time() - before
        print(name, ":", int(duration), "seconds")
        score = metric.average()
        total = metric.total()
        msg = "%s : avg=%.2f (total: %i)" % (name, score, total)
        print(msg)
        today = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        with open(filename, "a") as f:
            f.write("%s: %s\n" % (today, msg))
    finally:
        pass

@contextmanager
def measure_time(title):
    try:
        before = time.time()
        yield
    finally:
        duration = time.time() - before
        print(title, "elasped time", duration, "seconds")

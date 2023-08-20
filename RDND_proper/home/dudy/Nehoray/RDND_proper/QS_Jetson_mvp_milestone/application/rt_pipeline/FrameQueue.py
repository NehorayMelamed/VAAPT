import numpy as np


class FrameQueue:
    def __init__(self, q_size=50):
        self.q_size = q_size
        self.queue = ['o'] * q_size # use list to avoid small copies with np arrays
        self._index = 0

    def add_frame(self, frame: np.array):
        self.queue[self._index] = frame
        self._index+=1

    def is_full(self) -> bool:
        return self._index == self.q_size

    def get_queue(self) -> np.array:
        return np.array(self.queue)

    def empty_queue(self):
        self.queue = ['o'] * self.q_size
        self._index = 0
import numpy as np
from numpy import ndarray
from typing import Iterable


class Tensor(object):
    def __init__ (self, name: str, dimensions: Iterable[int]) -> None:
        self.name: str = name
        self.data: ndarray = np.random.random(dimensions) - 0.5
        self.diff: ndarray = np.zeros(dimensions)

# Copyright (C) Kamaledin Ghiasi-Shirazi, Ferdowsi Univerity of Mashhad, 2018 (1397 Hijri Shamsi)
#
# Author: 	Kamaledin Ghiasi-Shirazi

import numpy as np


class Tensor():
    def __init__ (self, name, dimensions):
        self.name = name
        self.data = np.random.random(dimensions) - 0.5
        self.diff = np.zeros(dimensions)

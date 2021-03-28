# Copyright (C) Kamaledin Ghiasi-Shirazi, Ferdowsi Univerity of Mashhad, 2018 (1397 Hijri Shamsi)
#
# Author: 	Kamaledin Ghiasi-Shirazi

import numpy as np


class Optimizer():
    def __init__ (self, params):
        self.params = params

    def zero_grad(self):
        for p in self.params:
            p.diff *= 0

    def step (self):
        pass


# Copyright (C) Kamaledin Ghiasi-Shirazi, Ferdowsi Univerity of Mashhad, 2018 (1397 Hijri Shamsi)
#
# Author: 	Kamaledin Ghiasi-Shirazi

from Optimizer import Optimizer


class SGD(Optimizer):
    def __init__ (self, params, lr):
        super().__init__(params)
        self.lr = lr

    def step (self):
        for p in self.params:
            p.data -= self.lr * p.diff

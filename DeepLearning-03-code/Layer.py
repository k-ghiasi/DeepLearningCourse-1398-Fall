# Copyright (C) Kamaledin Ghiasi-Shirazi, Ferdowsi Univerity of Mashhad, 2018 (1397 Hijri Shamsi)
#
# Author: 	Kamaledin Ghiasi-Shirazi

from Tensor import Tensor

class Layer():
    def __init__(self, layerName):
        self.name = layerName
        self.input_tensors_names = []
        self.output_tensors_names = []
        self.output_tensors = {}
        self.parameters = {}

    def setup(self, input_tensors):
        pass

    def forward(self, input_tensors):
        pass

    def backward(self, output_tensors):
        pass

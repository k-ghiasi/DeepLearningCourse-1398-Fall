# Copyright (C) Kamaledin Ghiasi-Shirazi, Ferdowsi Univerity of Mashhad, 2018 (1397 Hijri Shamsi)
#
# Author: 	Kamaledin Ghiasi-Shirazi

from Tensor import Tensor
from Layer import Layer

class Net():
    def __init__(self):
        self.tensors = {}
        self.layers = []

    def setup(self, inputs):
        for input in inputs:
            self.tensors[input.name] = input
        for layer in self.layers:
            input_tensors = [self.tensors[key]  for key in layer.input_tensors_names]
            layer.setup(input_tensors)
            for tensor_name in layer.output_tensors:
                self.tensors[tensor_name] = layer.output_tensors[tensor_name]

    def forward(self, inputs):
        for input in inputs:
            self.tensors[input.name] = input
        for layer in self.layers:
            input_tensors = [self.tensors[key]  for key in layer.input_tensors_names]
            layer.forward(input_tensors)

    def backward(self):
        for layer in reversed(self.layers):
            output_tensors = [self.tensors[key] for key in layer.output_tensors_names]
            layer.backward(output_tensors)

    def parameters(self):
        L = [layer.parameters[key]  for layer in self.layers for key in layer.parameters]
        return L

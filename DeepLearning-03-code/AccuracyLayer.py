# Copyright (C) Kamaledin Ghiasi-Shirazi, Ferdowsi Univerity of Mashhad, 2018 (1397 Hijri Shamsi)
#
# Author: 	Kamaledin Ghiasi-Shirazi

import numpy as np
from Layer import Layer
from Tensor import Tensor

class AccuracyLayer(Layer):

    def __init__(self, name):
        super().__init__(name)

    def setup(self, input_tensors):
        name = self.output_tensors_names[0]
        self.batch_size = input_tensors[0].data.shape[0]
        self.output_tensors[name] = Tensor(
            name=name, dimensions=[self.batch_size, 1])

    def forward(self, input_tensors):
        z = input_tensors[0].data
        y = np.reshape(np.argmax(z, axis=1), (z.shape[0],1))
        target = input_tensors[1].data
        out_tensor = self.output_tensors[self.output_tensors_names[0]]
        out_tensor.data = y == target

    def backward(self, output_tensors):
        pass

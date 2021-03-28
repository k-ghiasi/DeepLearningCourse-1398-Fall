# Copyright (C) Kamaledin Ghiasi-Shirazi, Ferdowsi Univerity of Mashhad, 2018 (1397 Hijri Shamsi)
#
# Author: 	Kamaledin Ghiasi-Shirazi

import numpy as np
from Tensor import Tensor
from Layer import Layer


class SoftMaxWithCrossEntropyLossLayer(Layer):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def setup(self, input_tensors):
        self.batch_size = input_tensors[0].data.shape[0]
        out_name = self.output_tensors_names[0]
        self.output_tensors[out_name] = Tensor(
            name=out_name, dimensions=(self.batch_size, 1))

    def forward(self, input_tensors):
        self.input_tensors = input_tensors
        z = input_tensors[0].data
        t = input_tensors[1].data
        z = z - np.max(z, axis=1).reshape([z.shape[0], 1])
        self.y = np.exp(z)
        sigma = np.sum(self.y, axis=1)
        sigma = np.reshape(sigma, [sigma.shape[0], 1])
        self.y = self.y / sigma
        chosen_y = np.choose(t.T, self.y.T)
        loss = -np.log(chosen_y + 1e-16)
        out_tensor = self.output_tensors[self.output_tensors_names[0]]
        out_tensor.data[:] = np.reshape(loss, (self.batch_size, 1))

    def backward(self, output_tensors):
        z_diff = self.input_tensors[0].diff
        target = self.input_tensors[1].data
        z_diff[:] = self.y
        for (i, t) in enumerate(target):
            z_diff[i, t] -= 1.0

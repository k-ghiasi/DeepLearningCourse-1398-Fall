# Copyright (C) Kamaledin Ghiasi-Shirazi, Ferdowsi Univerity of Mashhad, 2018 (1397 Hijri Shamsi)
#
# Author:     Kamaledin Ghiasi-Shirazi

import numpy as np
from Layer import Layer
from Tensor import Tensor


class InnerProductLayer(Layer):
    def __init__(self, name, output_dim):
        super().__init__(name)
        self.output_dim = output_dim

    def setup(self, input_tensors):
        self.input_tensors = input_tensors
        input_data = input_tensors[0]
        self.batch_size, self.input_dim = input_data.data.shape
        self.output_tensors[self.output_tensors_names[0]] = Tensor(
            name=self.output_tensors_names[0],
            dimensions=[self.batch_size, self.output_dim])
        self.parameters['W'] = Tensor(
            name='W', dimensions=[self.input_dim, self.output_dim])
        self.parameters['W'].data /= np.sqrt(self.input_dim)
        self.parameters['b'] = Tensor(
            name='b', dimensions=[self.output_dim])

    def forward(self, input_tensors):
        input_data = input_tensors[0]
        self.output_tensors[self.output_tensors_names[0]].data = \
            input_data.data @ self.parameters['W'].data \
                + self.parameters['b'].data

    def backward(self, output_tensors):
        x_diff = self.input_tensors[0].diff
        x_diff[:] = output_tensors[0].diff @ self.parameters['W'].data.T

        x = self.input_tensors[0].data
        self.parameters['W'].diff = x.T @ output_tensors[0].diff

        bias_x = np.ones([self.batch_size])
        self.parameters['b'].diff = bias_x.T @ output_tensors[0].diff

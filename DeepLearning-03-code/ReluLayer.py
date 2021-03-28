# Copyright (C) Kamaledin Ghiasi-Shirazi, Ferdowsi Univerity of Mashhad, 2018 (1397 Hijri Shamsi)
#
# Author: 	Kamaledin Ghiasi-Shirazi

import numpy as np
from Layer import Layer
from Tensor import Tensor

class ReluLayer(Layer):
    def __init__(self, name):
        super().__init__(name)

    def setup(self, input_tensors):
        name = self.output_tensors_names[0]
        self.output_tensors[name] = Tensor(
            name=name, dimensions= input_tensors[0].data.shape)

    def forward(self, input_tensors):
        self.input_tensors = input_tensors
        out_tensor = self.output_tensors[self.output_tensors_names[0]]
        out_tensor.data[:,:] = np.maximum(input_tensors[0].data, 0)

    def backward(self, output_tensors):
        do_di = self.input_tensors[0].data > 0.0
        dL_do = output_tensors[0].diff
        self.input_tensors[0].diff[:] = do_di * dL_do


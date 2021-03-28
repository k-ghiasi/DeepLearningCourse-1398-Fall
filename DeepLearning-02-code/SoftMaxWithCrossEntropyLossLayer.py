import numpy as np
from numpy import ndarray
from typing import List, Dict
from Tensor import Tensor
from Layer import Layer


class SoftMaxWithCrossEntropyLossLayer(Layer):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.input_tensors_names: List[str] = []
        self.output_tensors_names: List[str] = []
        self.output_tensors: Dict[str, Tensor] = {}
        self.parameters: Dict[str, Tensor] = {}

    def setup(self, input_tensors):
        nm = self.output_tensors_names[0]
        self.output_tensors[nm] = Tensor(
            name=nm, dimensions=[input_tensors[0].data.shape[0], 1])

    def forward(self, input_tensors):
        self.input_tensors = input_tensors
        z: ndarray = input_tensors[0].data
        target: ndarray = input_tensors[1].data
        z = z - np.max(z, axis=1).reshape([z.shape[0], 1])
        self.y = np.exp(z)
        sigma = np.sum(self.y, axis=1)
        sigma = np.reshape(sigma, [sigma.shape[0], 1])
        self.y = np.divide(self.y, sigma)
        chosen_y = np.choose(target.T, self.y.T)
        loss = -np.log(10 ** (-16) + chosen_y)
        self.output_tensors[self.output_tensors_names[0]].data[:] =\
			np.reshape(loss, [loss.shape[1], 1])

    def backward(self, output_tensors):
        z_diff: ndarray = self.input_tensors[0].diff
        target: ndarray = self.input_tensors[1].data
        z_diff[:] = self.y
        for (i, t) in enumerate(target):
            z_diff[i, t] -= 1.0

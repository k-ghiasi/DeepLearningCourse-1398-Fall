import numpy as np
from numpy import ndarray
from Layer import Layer
from Tensor import Tensor

class ReluLayer(Layer):

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def setup(self, input_tensors):
        name = self.output_tensors_names[0]
        self.output_tensors[name] = Tensor(
            name=name, dimensions= input_tensors[0].data.shape)

    def forward(self, input_tensors):
        z: ndarray = input_tensors[0].data
        self.input_tensors = input_tensors
        self.output_tensors[self.output_tensors_names[0]].data[:,:] =\
            np.maximum(input_tensors[0].data,0)

    def backward(self, output_tensors):
        po_pi = self.input_tensors[0].data > 0.0
        pL_po = output_tensors[0].diff
        self.input_tensors[0].diff[:] = po_pi * pL_po


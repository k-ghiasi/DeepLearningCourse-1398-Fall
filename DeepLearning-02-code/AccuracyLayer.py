import numpy as np
from numpy import ndarray
from Layer import Layer
from Tensor import Tensor

class AccuracyLayer(Layer):

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def setup(self, input_tensors):
        name = self.output_tensors_names[0]
        self.output_tensors[name] = Tensor(
            name=name, dimensions=[input_tensors[0].data.shape[0], 1])

    def forward(self, input_tensors):
        z: ndarray = input_tensors[0].data
        target: ndarray = input_tensors[1].data
        y = np.argmax(z, axis=1)
        true_predictions = y == target.squeeze()
        self.output_tensors[self.output_tensors_names[0]].data =\
            np.reshape(true_predictions, [z.shape[0], 1])

    def backward(self, output_tensors):
        pass

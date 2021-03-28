import numpy as np
from numpy import ndarray
from Layer import Layer
from Tensor import Tensor


class InnerProductLayer(Layer):

    def __init__(self, name: str, output_neurons: int) -> None:
        super().__init__(name)
        self.output_neurons = output_neurons

    def setup(self, input_tensors) -> None:
        input_data = input_tensors[0]
        batch_size: int = input_data.data.shape[0]
        self.output_tensors[self.output_tensors_names[0]] = Tensor(
            name=self.output_tensors_names[0],
            dimensions=[batch_size, self.output_neurons])
        self.parameters['W'] = Tensor(
            name='W', dimensions=[input_data.data.shape[1], self.output_neurons])
        self.parameters['W'].data /= \
            np.sqrt(input_data.data.shape[1])
        self.parameters['b'] = Tensor(
            name='b', dimensions=[self.output_neurons])

    def forward(self, input_tensors):
        self.input_tensors = input_tensors
        input_data = input_tensors[0]
        self.output_tensors[self.output_tensors_names[0]].data = \
            np.dot(input_data.data, self.parameters['W'].data) \
            + self.parameters['b'].data

    def backward(self, output_tensors):
        x_diff: ndarray = self.input_tensors[0].diff
        x_diff[:] = np.dot(output_tensors[0].diff,
                           self.parameters['W'].data.T)

        self.parameters['W'].diff = \
            np.dot(self.input_tensors[0].data.T, \
                   output_tensors[0].diff)

        bias_input = np.ones([output_tensors[0].data.shape[0], 1])
        self.parameters['b'].diff = \
            np.dot(bias_input.T, output_tensors[0].diff)

            #np.tensordot(
            #self.input_tensors[0].data, output_tensors[0].diff, axes=(0, 0))

        #self.parameters['b'].diff = np.tensordot(
        #    np.ones([output_tensors[0].data.shape[0], 1]),\
        #    output_tensors[0].diff, axes=(0, 0)).squeeze()



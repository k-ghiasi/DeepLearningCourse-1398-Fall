from typing import List, Dict
from Tensor import Tensor

class Layer(object):
    def __init__(self, name: str) -> None:
        self.name = name
        self.input_tensors_names: List[str] = []
        self.output_tensors_names: List[str] = []
        self.output_tensors: Dict[str, Tensor] = {}
        self.parameters: Dict[str, Tensor] = {}

    def setup(self, input_tensors):
        pass

    def forward(self, input_tensors):
        pass

    def backward(self, output_tensors):
        pass

import numpy as np
from dezero import Function
from dezero import as_variable

class Sin(Function):
    def forward(self, x):
        return np.sin(x)
    def backward(self, gy):
        x, = self.inputs
        return gy * cos(x)

def sin(x):
    return Sin()(x)

class Cos(Function):
    def forward(self, x):
        return np.cos(x)
    def backward(self, gy):
        x, = self.inputs
        return gy * -sin(x)

def cos(x):
    return Cos()(x)

class Tanh(Function):
    def forward(self, x):
        return np.tanh(x)
    def backward(self, gy):
        y = self.outputs[0]()
        return g * (1 - y*y)

def tanh(x):
    return Tanh()(x)

'''
class Sum(Function):
    def forward(self, x):
        self.shape = x.shape
        return np.sum(x.data, keepdims=True)
    def backward(self, gy):
        return reshape(gy, self.shape)

def sum(x):
    return Sum()(x)
'''

class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape
    def forward(self, x):
        self.x_shape = x.shape
        return x.reshape(self.shape)
    def backward(self, gy):
        return reshape(gy, self.x_shape)

def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)
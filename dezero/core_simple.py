import numpy as np
import heapq as hq
import weakref
import contextlib

class Config:
    enable_backprop = True

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def numerical_diff(f, x, eps=1e-4):
    return (f(Variable(x.data+eps)).data - f(Variable(x.data-eps)).data) / (2*eps)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

def no_grad():
    return using_config('enable_backprop', False)

class Variable:
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f'{type(data)}는 지원하지 않습니다.')
        self.name = name
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace("\\n", "\\n"+" "*9)
        return f'variable({p})'

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        seen_set = set()
        funcs = []

        def add_func(f):
            if f not in seen_set:
                seen_set.add(f)
                hq.heappush(funcs, f)
        
        add_func(self.creator)

        while funcs:
            f = hq.heappop(funcs)
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs, )
            for x, gx in zip(f.inputs, gxs):
                x.grad = gx if x.grad is None else gx + x.grad
                if x.creator is not None:
                    add_func(x.creator)
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys, )
        outputs = [Variable(as_array(y)) for y in ys]
        
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

            return outputs if len(outputs) > 1 else outputs[0]

    def __lt__(self, other): # this for generation heap
        if self.generation > other.generation:
            return True
        return False

    def __str__(self):
        return f'{self.generation} gen : [{__class__}], {id(self)}'

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, x):
        raise NotImplementedError()

class Neg(Function):
    def forward(self, x):
        return -1 * x
    def backward(self, gy):
        return -1*gy

class Pow(Function):
    def __init__(self, c):
        self.c = c
    def forward(self, x):
        return x**self.c
    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c
        return gy*c*x**(c-1)

class Square(Function):
    def forward(self, x):
        return x**2
    def backward(self, gy):
        x = self.inputs[0].data
        return 2 * x * gy

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    def backward(self, gy):
        return np.exp(self.inputs.data) * gy

class Add(Function):
    def forward(self, x0, x1):
        return x0 + x1
    def backward(self, gy):
        return gy, gy

class Mul(Function):
    def forward(self, x0, x1):
        return x0 * x1
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return x1*gy, x0*gy

class Sub(Function):
    def forward(self, x0, x1):
        return x0 - x1
    def backward(self, gy):
        return gy, -gy

class Div(Function):
    def forward(self, x0, x1):
        return x0 / x1
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy/x1, (-x0/x1**2)*gy

def neg(x):
    return Neg()(x)
def pow(x, c):
    return Pow(c)(x)
def square(x):
    return Square()(x)
def exp(x):
    return Exp()(x)
def add(x, y):
    y = as_array(y)
    return Add()(x, y)
def mul(x, y):
    y = as_array(y)
    return Mul()(x, y)
def sub(x, y):
    y = as_array(y)
    return Sub()(x, y)
def rsub(x, y):
    y = as_array(y)
    return Sub()(y, x)
def div(x, y):
    y = as_array(y)
    return Div()(x, y)
def rdiv(x, y):
    y = as_array(y)
    return Div()(y, x)

def setup_variable():
    Variable.__neg__ = neg
    Variable.__pow__ = pow
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
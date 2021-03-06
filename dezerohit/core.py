import weakref
import numpy as np
import contextlib
import dezerohit
import heapq

# ==================================================
# Config
# ==================================================


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


def no_grad():
    return using_config("enable_backprop", False)


# ==================================================
# Variable / Function
# ==================================================


class Variable:
    # Variable should have priority about calling magic method, when operated with ndarray instance
    __array_priority__ = 200  

    def __init__(self, data, name=None):
        if data is not None:
            # only ndarray
            if not isinstance(data, np.ndarray):
                raise TypeError("Can't read {}.".format(type(data)))

        self.data = data
        self.name = name  # for visualization of computational graph
        self.grad = None
        self.creator = None  # Func as creator created this instance.
        self.generation = 0  # The later instance is created, the bigger generation is.(for topology)

    # property tag makes us use method like instance variable.
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    # Ndarray instance is initialized in float64 or int64 type, but in neural network usually we use float32.
    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return "variable(None)"
        p = str(self.data).replace("\n", "\n" + " " * 9)
        return "variable(" + p + ")"

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1  # generation of output variable

    def cleargrad(self):  # reuse instance
        self.grad = None

    def backward(self, retain_grad=False, create_graph=False):
        """
        retain_grad: if False the grad of y(dy) is removed after operating dy/dx.
        create_graph: if True dezerohit can do high-order differentiation, else only first order.
        """

        if self.grad is None:  # if this variable is last output,
            self.grad = Variable(np.ones_like(self.data))  # ones_like: for same data type

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                heapq.heappush(funcs, (-f.generation, id(f), f)) # if f.gen is not unique, compare id of func object. 
                seen_set.add(f)

        add_func(self.creator)

        while funcs:
            f = heapq.heappop(funcs)[2]
            gys = [output().grad for output in f.outputs]  # Output is weakref.

            # f.backward(*gys) and gradient addition call Function().__call__ and refer Config.enable_backprop
            with using_config("enable_backprop", create_graph):
                gxs = f.backward(*gys)  # unpacking
                if not isinstance(gxs, tuple):  # Type of outputs should be tuple or list.
                    gxs = (gxs,)

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx  # Don't use compound assignments which are in-place operation in numpy.

                    if x.creator is not None:
                        add_func(x.creator)

            if not retain_grad:  # Remove dy after operating dy/dx.
                for y in f.outputs:
                    y().grad = None  # y is weakref.

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return dezerohit.functions.reshape(self, shape)  # Note dezerohit.functions.reshape: avoid circular imports

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:  # tuple or list or None
                axes = axes[0]
        return dezerohit.functions.transpose(self, axes)  # avoid circular imports

    @property
    def T(self):
        return dezerohit.functions.transpose(self)

    def sum(self, axis=None, keepdims=False):
        return dezerohit.functions.sum(self, axis, keepdims)


class Parameter(Variable):
    pass


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, *inputs):
        """
        Note: weakref.ref(output)

        Function instance refers list of Variable instances as self.outputs.
        List of Variable instances also refers Function instance as self.creator.
        Therefore there is a circular reference between them, which make them be left until GC(garbage collector) reclaim memory.

        The solution to break up this circular reference is that Function instance refers list of Variable instances weakly 
        so as to keep output's reference count.
        As we use weakref, their reference counts changed from 2 to 1.
        
        After we stop referring list of Variable instances(outputs), its reference count should be zero and it'll be reclaimed by GC.
        So as outputs are freed, Function instance's reference count also should be zero and it'll be freed too.
        """

        # for operation between Variable instance and ndarray instance
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]

        ys = self.forward(*xs)  # unpacking

        if not isinstance(ys, tuple):  # Type of outputs should be tuple or list.
            ys = (ys,)

        # Note as_array(y): if x.ndim == 0, type(y) != numpy.ndarray.
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:  # backward activation toggle(default: true)
            # Generation of func is Generation of youngest input variable.(searching topology)
            self.generation = max([x.generation for x in inputs])

            for output in outputs:
                output.set_creator(self)

            self.inputs = inputs
            # Output's reference count doesn't be increased.
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


# ==================================================
# Arithmetic operations / Operator overload
# ==================================================


class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y

    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:  # for broadcast
            gx0 = dezerohit.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezerohit.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def add(x0, x1):
    # Ndarray instance 'x1' will be transformed to variable obj at Function call.
    x1 = as_array(x1)
    return Add()(x0, x1)


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy * x1
        gx1 = gy * x0
        if x0.shape != x1.shape:  # for broadcast
            gx0 = dezerohit.functions.sum_to(gx0, x0.shape)
            gx1 = dezerohit.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


def mul(x0, x1):
    # Ndarray instance 'x1' will be transformed to variable obj at Function call.
    x1 = as_array(x1)
    return Mul()(x0, x1)


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


def neg(x):
    return Neg()(x)


class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return y

    def backward(self, gy):
        gx0 = gy
        gx1 = -gy
        if self.x0_shape != self.x1_shape:  # for broadcast
            gx0 = dezerohit.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezerohit.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):  # x0=self, x1=other type
    x1 = as_array(x1)
    return Sub()(x1, x0)  # The order of args should be changed.


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        if x0.shape != x1.shape:  # for broadcast
            gx0 = dezerohit.functions.sum_to(gx0, x0.shape)
            gx1 = dezerohit.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):  # x0=self, x1=other type
    x1 = as_array(x1)
    return Div()(x1, x0)  # The order of args should be changed.


class Pow(Function):  # Assume c is constant, so there's no need to get gy/gc.
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        (x,) = self.inputs
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx


def pow(x, c):
    return Pow(c)(x)


def setup_variable():
    """
    operator overload

    ex)
    
    [ y = add(mul(a, b), c) ] => [ y = a * b + c ]
    """
    Variable.__add__ = add
    Variable.__mul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__pow__ = pow
    Variable.__truediv__ = div
    Variable.__getitem__ = dezerohit.functions.get_item

    # operation with other type
    # the order of args doesn't matter.
    Variable.__radd__ = add
    Variable.__rmul__ = mul
    Variable.__rsub__ = rsub
    Variable.__rtruediv__ = rdiv


import contextlib
import numpy as np
import unittest
import weakref

import dezero


class Config:
    enable_backprop = True


class Variable:
    __array_proority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{} is not suppported".format(type(data)))

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad=False, create_graph=False):
        """反向传播过程

        retain_grad: 是否保留中间过程的grad, 默认为False不保存
        create_graph: 是否创建反向传播计算图, 默认为False不创建
        """

        if self.grad is None:
            # self.grad = np.ones_like(self.data)
            self.grad = Variable(np.ones_like(self.data))

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            # 开始反向传播计算
            gys = [
                output().grad for output in f.outputs
            ]  # 将Variable的实例变量grad汇总在列表中

            with using_config(
                "enable_backprop", create_graph
            ):  # 配合Function::forward中的`if Config.enable_backprop:`来创建反向连接
                gxs = f.backward(
                    *gys
                )  # 进行实际的反向传播运算，在前向过程的输出就是后向过程的输入
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)
                for x, gx in zip(
                    f.inputs, gxs
                ):  # 从输出端开始传播的导数（gx）设置魏函数的输入变量（f.input）的grad
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx  # 累积多个梯度

                    if x.creator is not None:
                        add_func(x.creator)

                if not retain_grad:  # 不保留中间梯度时，要清空所有已经用过的grad
                    for y in f.outputs:
                        y().grad = None

    def cleargrad(self):
        self.grad = None

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

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

    # def __mul__(self, other):
    #     return mul(self, other)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return dezero.functions.reshape(self, shape)

    def transpose(self):
        return dezero.functions.transpose(self)

    @property
    def T(self):
        return dezero.functions.transpose(self)

    def sum(self, axis=None, keepdims=False):
        return dezero.functions.sum(self, axis, keepdims)


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
        # 正向传播的计算
        xs = [x.data for x in inputs]  # 提取Variable的实例变量data并汇总到列表xs中
        ys = self.forward(*xs)  # 实际执行forward方法进行前向计算
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max(
                [x.generation for x in inputs]
            )  # 获取当前最大的层级树保证能够完成反向传播图中按照正确的顺序进行反向传播
            # 创建连接
            for output in outputs:
                output.set_creator(self)  # 输出变量保存创造者信息
            self.inputs = inputs  # 保存输入的变量
            self.outputs = [
                weakref.ref(output) for output in outputs
            ]  # 保存输入的变量，通过弱引用来解除循环引用

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, xs):
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y

    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        # x0, x1 = self.inputs[0].data, self.inputs[1].data  # 之前的实现时从Variable中取出数据（ndarray实例）
        x0, x1 = self.inputs
        return (
            gy * x1,
            gy * x0,
        )  # 因为现在x1、x0和gy都是Variable，所以会调用mul继续创建计算图


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        return x0 - x1

    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, -gx1


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        # x0, x1 = self.inputs[0].data, self.inputs[1].data
        x0, x1 = self.inputs
        gx0 = gy / 1
        gx1 = gy * (-x0 / x1**2)
        return gx0, gx1


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x**self.c
        return y

    def backward(self, gy):
        # x = self.inputs[0].data
        x = self.inputs[0]
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx


def pow(x, c):
    return Pow(c)(x)


def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)


def neg(x):
    return Neg()(x)


def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)


def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)


def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


import contextlib


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

import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func
        
    def backward(self):
        f = self.creator
        if f is not None:
            x = f.input
            x.grad = f.backward(self.grad)
            x.backward()


class Function:
    def __call__(self, input):
        x = input.data  # 取出数据
        y = self.forward(x)  # 实际的计算
        output = Variable(y)  # 作为Variable返回
        output.set_creator(self)  # 输出变量保存创造者信息
        self.input = input  # 保存输入的变量
        self.output = output  # 保存输出变量 
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, x):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x**2

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy  # 通过这个参数传播来的导数和y=x^2的导数的乘积
        return gx  # 返回结果会进一步向输入方向传播


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

y.grad = np.array(1.0)
y.backward()
print(x.grad)

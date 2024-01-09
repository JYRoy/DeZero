import numpy as np


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not suppported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func
        
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)
            
            if x.creator is not None:
                funcs.append(x.creator)


class Function:
    def __call__(self, input):
        x = input.data  # 取出数据
        y = self.forward(x)  # 实际的计算
        output = Variable(as_array(y))  # 作为Variable返回
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
    
def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

x = Variable(np.array(0.5))
y = square(exp(square(x)))

y.backward()
print(x.grad)

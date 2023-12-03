import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data

class Function:
    def __call__(self, input):
        x = input.data        # 取出数据
        y = self.forward(x)   # 实际的计算
        output = Variable(y)  # 作为Variable返回
        return output

    def forward(self, x):
        raise NotImplementedError()
    
class Square(Function):
    def forward(self, x):
        return x ** 2
    
x = Variable(np.array(10))
f = Square()
y = f(x)

print(type(x))
print(y.data)
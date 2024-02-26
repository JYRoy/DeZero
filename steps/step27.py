if (
    "__file__" in globals()
):  # 使用pip install dezero安装时DeZero包将会被配置在Python搜索路径中
    # 使用google colab时或者使用Python解释器的交互模式下，__file__变量没有被定义
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero import Variable, sin
import math

def my_sin(x, threshold=0.0001):
    y = 0
    for i in range(100000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y

x0 = Variable(np.array(np.pi/4))
y0 = my_sin(x0)
y0.backward()
print(y0.data)
print(x0.grad)

x1 = Variable(np.array(np.pi/4))
y1 = sin(x1)
y1.backward()
print(y1.data)
print(x1.grad)
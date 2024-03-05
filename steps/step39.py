if (
    "__file__" in globals()
):  # 使用pip install dezero安装时DeZero包将会被配置在Python搜索路径中
    # 使用google colab时或者使用Python解释器的交互模式下，__file__变量没有被定义
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
from dezero.utils import plot_dot_graph
import dezero.functions as F


# 一维数组
# x = Variable(np.array([[1, 2, 3, 4, 5, 6]]))
# y = F.sum(x)
# y.backward()
# print(y)
# print(x.grad)

# 二维数组
x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.sum(x)
y.backward()
print(y)
print(x.grad)

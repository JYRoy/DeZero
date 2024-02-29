if (
    "__file__" in globals()
):  # 使用pip install dezero安装时DeZero包将会被配置在Python搜索路径中
    # 使用google colab时或者使用Python解释器的交互模式下，__file__变量没有被定义
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
import dezero.functions as F

x = Variable(np.linspace(-7, 7, 200))
y = F.sin(x)
y.backward(create_graph=True)

logs = [y.data]

for i in range(3):
    logs.append(x.grad.data)
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)

labels = ["y=sin(x)", "y'", "y''", "y'''"]
for i, v in enumerate(logs):
    plt.plot(x.data, logs[i], label=labels[i])

plt.legend(loc='lower right')
# plt.show()
plt.savefig("./tmp.png")

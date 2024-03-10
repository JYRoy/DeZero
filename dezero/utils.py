import os
import subprocess


def _dot_var(v, verbose=False):
    dot_var = '{} [label="{}", color=orange, style=filled]\n'

    name = "" if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ": "
        name += str(v.shape) + " " + str(v.dtype)
    return dot_var.format(id(v), name)


def _dot_func(f):
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]'
    txt = dot_func.format(id(f), f.__class__.__name__)
    dot_edge = "{} -> {}\n"
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y()))  # y是weakref
    return txt


def get_dot_graph(output, verbose=True):
    txt = ""
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose)

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)

            if x.creator is not None:
                add_func(x.creator)

    return "digraph g {\n" + txt + "}"


def plot_dot_graph(output, verbose=True, to_file="graph.png"):
    dot_graph = get_dot_graph(output, verbose)

    tmp_dir = os.path.join(os.path.expanduser("~"), ".dezero")
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    graph_path = os.path.join(tmp_dir, "tmp_graph.dot")

    with open(graph_path, "w") as f:
        f.write(dot_graph)

    extension = os.path.splitext(to_file)[1][1:]
    cmd = "dot {} -T {} -o {}".format(graph_path, extension, to_file)
    subprocess.run(cmd, shell=True)


def sum_to(x, shape):
    """求x的元素之和，并将结果的形状转变为shape的形状

    x -- 多维数组
    shape -- 目标形状的数组
    """
    ndim = len(shape)
    lead = x.ndim - ndim  # lead表示需要压缩的维度数量
    lead_axis = tuple(range(lead))  # 从0开始的整数，表示要压缩的维度

    axis = tuple(
        [i + lead for i, sx in enumerate(shape) if sx == 1]
    )  # 如果某个维度的长度为1，则将该维度的索引加上lead后加入到axis数组中，axis表示在x上进行求和操作的维度
    y = x.sum(lead_axis + axis, keepdims=True)  # 在lead_axis + axis指定的维度上进行求和
    if lead > 0:
        y = y.squeeze(lead_axis)  # 压缩lead_axis指定的维度
    return y

def reshape_sum_backward(gy, shape, axis, keepdims):
    pivot_shape: tuple = (shape[0], 1)

    if not gy.shape:
        gy = gy.reshape((1,))

    if axis == 0:
        return gy.tile(pivot_shape)
    elif axis == 1:
        return gy.reshape(pivot_shape)
    else: # when axis is None
        return gy
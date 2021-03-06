import os
import subprocess
import urllib.request
import numpy as np
from dezerohit import as_variable
from dezerohit import Variable
# from dezerohit import cuda

# ===========================================================
# Visualize for computational graph
# ===========================================================

# transform variable instance into DOT language
def _dot_var(v, verbose=False): # verbose: print shape and type of instance
    dot_var = '{} [label="{}", color=orange, style=filled]\n'

    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' + str(v.dtype)
    return dot_var.format(id(v), name)

# transform function instance into DOT language
def _dot_func(f):
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    ret = dot_func.format(id(f), f.__class__.__name__)

    dot_edge = '{} -> {}\n'

    for x in f.inputs:
        ret += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        ret += dot_edge.format(id(f), id(y())) # y is weak reference.
    return ret

# write DOT texts tracking backward path of last output.
def get_dot_graph(output, verbose=True):
    """Generates a graphviz DOT text of a computational graph.
    Build a graph of functions and variables backward-reachable from the
    output. To visualize a graphviz DOT text, you need the dot binary from the
    graphviz package (www.graphviz.org).
    Args:
        output (dezero.Variable): Output variable from which the graph is
            constructed.
        verbose (bool): If True the dot graph contains additional information
            such as shapes and dtypes.
    Returns:
        str: A graphviz DOT text consisting of nodes and edges that are
            backward-reachable from the output
    """

    txt = ''
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f) # unlike backward(), no need to sort funcs 
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
    
    return 'digraph g {\n' + txt + '}'

def plot_dot_graph(output, verbose=True, to_file='graph.jpg'):
    '''verbose (bool): If True the dot graph contains additional information
            such as shapes and dtypes.
    '''
    
    dot_graph = get_dot_graph(output, verbose)

    tmp_dir = os.path.join(os.path.expanduser('~'), '.dezerohit')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')

    with open(graph_path, 'w') as f:
        f.write(dot_graph)
    
    extension = os.path.splitext(to_file)[1][1:] # Extension(e.g. png, pdf)
    cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, to_file)
    subprocess.run(cmd, shell=True)

    # Return the image as a Jupyter Image object, to be displayed in-line.
    try:
        from IPython import display
        return display.Image(filename=to_file)
    except:
        pass

# ===========================================================
# Utility functions for numpy (numpy magic)
# ===========================================================

def sum_to(x, shape):
    """Sum elements along axes to output an array of a given shape.
    Args:
        x (ndarray): Input array.
        shape:
    Returns:
        ndarray: Output array of the shape.
    """

    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead)) # used at y.squeeze

    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1]) # Add axis along which input array is summed.
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y

def reshape_sum_backward(gy, x_shape, axis, keepdims):
    """Reshape gradient appropriately for dezerohit.functions.sum's backward.
    Args:
        gy (dezero.Variable): Gradient variable from the output by backprop.
        x_shape (tuple): Shape used at sum function's forward.
        axis (None or int or tuple of ints): Axis used at sum function's
            forward.
        keepdims (bool): Keepdims used at sum function's forward.
    Returns:
        dezero.Variable: Gradient variable which is reshaped appropriately
    """

    ndim = len(x_shape)
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)
    
    if not (ndim == 0 or tupled_axis is None or keepdims): # if input tensor are summed by axes and its dims are changed at sum's forward,
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis] # Note a + ndim: a == -1
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1) # Input tensor's dims are recovered.
    else:
        shape = gy.shape
    
    gy = gy.reshape(shape)
    return gy

# ===========================================================
# Gradient check
# ===========================================================

def gradient_check(f, x, *args, rtol=1e-4, atol=1e-5, **kwargs):
    """Test backward procedure of a given function.
    This automatically checks the backward-process of a given function. For
    checking the correctness, this function compares gradients by
    backprop and ones by numerical derivation. If the result is within a
    tolerance this function return True, otherwise False.
    Args:
        f (callable): A function which gets `Variable`s and returns `Variable`s.
        x (`ndarray` or `dezero.Variable`): A traget `Variable` for computing
            the gradient.
        *args: If `f` needs variables except `x`, you can specify with this
            argument.
        rtol (float): The relative tolerance parameter.
        atol (float): The absolute tolerance parameter.
        **kwargs: If `f` needs keyword variables, you can specify with this
            argument.
    Returns:
        bool: Return True if the result is within a tolerance, otherwise False.
    """
    x = as_variable(x)
    x.data = x.data.astype(np.float64)

    num_grad = numerical_grad(f, x, *args, **kwargs)
    y = f(x, *args, **kwargs)
    y.backward()
    bp_grad = x.grad.data

    assert bp_grad.shape == num_grad.shape
    res = np.allclose(x.grad.data, num_grad, atol=atol, rtol=rtol)

    if not res:
        print('')
        print('========== FAILED (Gradient Check) ==========')

        print('Numerical Grad')
        print(f' shape: {num_grad.shape}')
        val = str(num_grad.flatten()[:10])
        print(f' values: {val[1:-1]} ...')

        print('Backprop Grad')
        print(f' shape: {x.grad.shape}')
        val = str(x.grad.data.flatten()[:10])
        print(f' values: {val[1:-1]} ...')
    return res

def numerical_grad(f, x, *args, **kwargs):
    """Computes numerical gradient by finite differences.
    Args:
        f (callable): A function which gets `Variable`s and returns `Variable`s.
        x (`ndarray` or `dezero.Variable`): A target `Variable` for computing
            the gradient.
        *args: If `f` needs variables except `x`, you can specify with this
            argument.
        **kwargs: If `f` needs keyword variables, you can specify with this
            argument.
    Returns:
        `ndarray`: Gradient.
    """
    eps = 1e-4

    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0, *args, **kwargs)
    y1 = f(x1, *args, **kwargs)
    return (y1.data - y0.data) / (2 * eps)
    


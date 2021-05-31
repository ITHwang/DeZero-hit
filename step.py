# 47

import numpy as np

# import matplotlib.pyplot as plt
# plt.switch_backend('agg')
from dezerohit import Variable, Model
import dezerohit.functions as F
from dezerohit.utils import plot_dot_graph, sum_to
import dezerohit.layers as L
from dezerohit.models import MLP
from dezerohit import optimizers

model = MLP((10, 3))
x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, 3.2], [2.1, 0.3]])
t = np.array([2, 0, 1, 0])
y = model(x)
loss = F.softmax_cross_entropy_simple(y, t)
print(loss)

#44

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from dezerohit import Variable, Model
import dezerohit.functions as F
from dezerohit.utils import plot_dot_graph, sum_to
import dezerohit.layers as L
from dezerohit.models import MLP
from dezerohit import optimizers

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

lr = 0.2
max_iter = 10000
hidden_size = 10

model = MLP((hidden_size, 1))
optimizer = optimizers.SGD(lr).setup(model)

for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    optimizer.update()
    if i % 1000 == 0:
        print(loss)

plt.scatter(x, y, s=20)
plt.xlabel('x')
plt.ylabel('y')
t = np.arange(0, 1, .01)[:, np.newaxis]
y_pred = model(t)
plt.plot(t, y_pred.data, color='r')
plt.savefig('model_train.png')






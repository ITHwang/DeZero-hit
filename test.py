import numpy as np
from dezerohit.models import MLP

model = MLP((10, 3))

x = np.array([[0.2, -0.4]])
y = model(x)
print(y)



import numpy as np
from dezerohit import Layer
import dezerohit.functions as F
import dezerohit.layers as L
from dezerohit import utils

# ========================================================
# Model / MLP
# ========================================================

class Model(Layer):
    def plot(self, *inputs, to_file='model.png'): # only forward graph
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)

# class Sequential(Model):

class MLP(Model): # Multi-Layer Perceptron
    def __init__(self, fc_output_sizes, activation=F.sigmoid):
        '''
        fc_output_sizes(tuple or list): output sizes of fully-connected layers.
        '''

        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, 'l' + str(i), layer) # Layer.__setattr__
            self.layers.append(layer)
        
    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)
            
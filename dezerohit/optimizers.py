import math
from dezerohit import Parameter # cuda

# ======================================================
# Optimizer (base class)
# ======================================================

class Optimizer:
    def __init__(self):
        self.target = None # target model whose params will be updated
        self.hooks = [] # preprocessors
    
    def setup(self, target): 
        self.target = target
        return self
    
    def update(self):
        params = [p for p in self.target.params() if p.grad is not None]

        for f in self.hooks:
            f(params)
        
        for param in params:
            self.update_one(param)
    
    def update_one(self, param):
        raise NotImplementedError()
    
    def add_hook(self, f):
        self.hooks.append(f)

# ======================================================
# Hook functions
# ======================================================

class WeightDecay:
    def __init__(self, rate):
        self.rate = rate
    
    def __call__(self, params):
        for param in params:
            param.grad.data += self.rate * param.data # W <- W - lr(dL/dW + beta*W)

class ClipGrad:
    def __init__(self, max_norm):
        self.max_norm = max_norm
    
    def __call__(self, params):
        total_norm = 0
        for param in params:
            total_norm += (param.grad.data ** 2).sum()
        total_norm = math.sqrt(float(total_norm))

        rate = self.max_norm / (total_norm + 1e-6)
        if rate < 1:
            for param in params:
                param.grad.data *= rate

# class FreezeParam:

# ======================================================
# SGD / 
# ======================================================

class SGD(Optimizer): # Stochastic Gradient Descent
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr
    
    def update_one(self, param):
        param.data -= self.lr * param.grad.data






            
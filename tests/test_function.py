if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
from numpy.core.numeric import array_equal
from dezerohit import Variable
from dezerohit.utils import gradient_check
import dezerohit.functions as F

class TestExp(unittest.TestCase):

    def test_exp_variable_forward(self):
        x = Variable(np.array([1, 2, 3]))
        y = F.exp(x)
        expected = np.exp(x.data)
        self.assertTrue(array_equal(y.data, expected))
    
    def test_exp_variable_backward1(self):
        x = Variable(np.random.randn(3, 3))
        self.assertTrue(gradient_check(F.exp, x))

    def test_exp_variable_backward2(self):
        x = Variable(np.random.randn(3))
        self.assertTrue(gradient_check(F.exp, x))

class TestSquare(unittest.TestCase):

    def test_square_variable_forward(self):
        x = Variable(np.array([1, 2, 3]))
        y = F.square(x)
        expected = np.array([1, 4, 9])
        self.assertTrue(array_equal(y.data, expected))
    
    def test_square_variable_backward1(self):
        x = Variable(np.random.randn(3, 3))
        self.assertTrue(gradient_check(F.square, x))

    def test_square_variable_backward2(self):
        x = Variable(np.random.randn(3))
        self.assertTrue(gradient_check(F.square, x))
import unittest
import numpy as np
from numpy.core.numeric import array_equal
from dezerohit import Variable
from dezerohit.utils import gradient_check
import dezerohit.functions as F
import dezerohit.core as Core

class TestAdd(unittest.TestCase):

    def test_forward1(self):
        x0 = np.array([1, 2, 3])
        x1 = Variable(np.array([1, 2, 3]))
        y = x0 + x1
        res = y.data
        expected = np.array([2, 4, 6])
        self.assertTrue(array_equal(res, expected))

    def test_backward1(self):
        x = Variable(np.random.randn(3, 3))
        y = np.random.randn(3, 3)
        self.assertTrue(gradient_check(Core.add, x, y))

    def test_backward2(self):
        x = Variable(np.random.randn(3, 3))
        y = np.random.randn(3, 1)
        self.assertTrue(gradient_check(Core.add, x, y))

    def test_backward3(self):
        x = np.random.randn(3, 3)
        y = np.random.randn(3, 1)
        self.assertTrue(gradient_check(Core.add, x, y))

class TestExp(unittest.TestCase):

    def test_forward1(self):
        x = Variable(np.array([1, 2, 3]))
        y = F.exp(x)
        expected = np.exp(x.data)
        self.assertTrue(array_equal(y.data, expected))
    
    def test_backward1(self):
        x = Variable(np.random.randn(3, 3))
        self.assertTrue(gradient_check(F.exp, x))

    def test_backward2(self):
        x = Variable(np.random.randn(3))
        self.assertTrue(gradient_check(F.exp, x))

class TestSquare(unittest.TestCase):

    def test_forward1(self):
        x = Variable(np.array([1, 2, 3]))
        y = F.square(x)
        expected = np.array([1, 4, 9])
        self.assertTrue(array_equal(y.data, expected))
    
    def test_backward1(self):
        x = Variable(np.random.randn(3, 3))
        self.assertTrue(gradient_check(F.square, x))

    def test_backward2(self):
        x = Variable(np.random.randn(3))
        self.assertTrue(gradient_check(F.square, x))


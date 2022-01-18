if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
from numpy.core.numeric import array_equal
from dezerohit import Variable
from dezerohit.utils import gradient_check
import dezerohit.functions as F

class TestNeg(unittest.TestCase):
    
    def test_change_sign_of_Variable_when_forward(self):
        x = Variable(np.array([1, 2, 3]))
        y = -x
        res = y.data
        expected = np.array([-1, -2, -3])
        self.assertTrue(array_equal(res, expected))
    
    def test_change_sign_of_numpy_when_forward(self):
        x = np.array([1, 2, 3])
        y = -x
        res = y.data
        expected = np.array([-1, -2, -3])
        self.assertTrue(array_equal(res, expected))
    
    def test_change_sign_of_Variable_when_backward(self):
        x = Variable(np.random.randn(3, 3))
        f = lambda x: -x
        self.assertTrue(gradient_check(f, x))

    def test_change_sign_of_numpy_when_backward(self):
        x = Variable(np.random.randn(5, 5))
        f = lambda x: -x
        self.assertTrue(gradient_check(f, x))

class TestAdd(unittest.TestCase):

    def test_add_ndarray_variable_forward(self):
        x0 = np.array([1, 2, 3])
        x1 = Variable(np.array([1, 2, 3]))
        y = x0 + x1
        res = y.data
        expected = np.array([2, 4, 6])
        self.assertTrue(array_equal(res, expected))

    def test_add_variable_ndarray_backward1(self):
        x = Variable(np.random.randn(3, 3))
        y = np.random.randn(3, 3)
        f = lambda x, y: x + y
        self.assertTrue(gradient_check(f, x, y))

    def test_add_variable_ndarray_backward2(self):
        x = Variable(np.random.randn(3, 3))
        y = np.random.randn(3, 1)
        f = lambda x, y: x + y
        self.assertTrue(gradient_check(f, x, y))

    def test_add_numpy_numpy_backward(self):
        x = np.random.randn(3, 3)
        y = np.random.randn(3, 1)
        f = lambda x, y: x + y
        self.assertTrue(gradient_check(f, x, y))

class TestMul(unittest.TestCase):

    def test_mul_ndarray_variable_forward(self):
        x0 = np.array([1, 2, 3])
        x1 = Variable(np.array([1, 2, 3]))
        y = x0 * x1
        res = y.data
        expected = np.array([1, 4, 9])
        self.assertTrue(array_equal(res, expected))

    def test_mul_ndarray_ndarray_backward1(self):
        x = np.random.randn(3, 3)
        y = np.random.randn(3, 3)
        f = lambda x, y: x * y
        self.assertTrue(gradient_check(f, x, y))

    def test_mul_ndarray_ndarray_backward2(self):
        x = np.random.randn(3, 3)
        y = np.random.randn(3, 1)
        f = lambda x, y: x * y
        self.assertTrue(gradient_check(f, x, y))

    def test_mul_ndarray_ndarray_backward3(self):
        x = np.random.randn(3, 3)
        y = np.random.randn(3, 1)
        f = lambda x, y: x * y
        self.assertTrue(gradient_check(f, x, y))

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

class TestSub(unittest.TestCase):

    def test_sub_variable_ndarray_forward(self):
        x0 = Variable(np.array([4, 5, 6]))
        x1 = np.array([1, 2, 3])
        result = (x0 - x1).data
        expected = np.array([3, 3, 3])
        return self.assertTrue(array_equal(expected, result))
    
    def test_sub_ndarray_variable_forward(self):
        x0 = np.array([1, 2, 3])
        x1 = Variable(np.array([4, 5, 6]))
        result = (x0 - x1).data
        expected = np.array([-3, -3, -3])
        return self.assertTrue(array_equal(expected, result))

    def test_sub_ndarray_variable_backward(self):
        x0 = np.random.randn(3, 3)
        x1 = Variable(np.random.randn(3, 3))
        f = lambda x, y: x - y    
        self.assertTrue(gradient_check(f, x0, x1))

    def test_sub_variable_ndarray_backward(self):
        x0 = Variable(np.random.randn(3, 3))
        x1 = np.random.randn(3, 3)
        f = lambda x, y: x - y    
        self.assertTrue(gradient_check(f, x0, x1))

class TestDiv(unittest.TestCase):

    def test_div_variable_ndarray_forward(self):
        x0 = Variable(np.array([4, 5, 6]))
        x1 = np.array([1, 2, 3])
        result = (x0 / x1).data
        expected = np.array([4/1, 5/2, 6/3])
        return self.assertTrue(array_equal(result, expected))
    
    def test_div_ndarray_variable_forward(self):
        x0 = np.array([1, 2, 3])
        x1 = Variable(np.array([4, 5, 6]))
        result = (x0 / x1).data
        expected = np.array([1/4, 2/5, 3/6])
        return self.assertTrue(array_equal(result, expected))

    def test_div_ndarray_variable_backward(self):
        x0 = np.random.randn(3, 3)
        x1 = Variable(np.random.randn(3, 3))
        f = lambda x, y: x / y    
        self.assertTrue(gradient_check(f, x0, x1))

    def test_div_variable_ndarray_backward(self):
        x0 = Variable(np.random.randn(3, 3))
        x1 = np.random.randn(3, 3)
        f = lambda x, y: x / y    
        self.assertTrue(gradient_check(f, x0, x1))

class TestPow(unittest.TestCase):

    def test_pow_variable_forward(self):
        x = Variable(np.array([4, 5, 6]))
        result = (x**2).data
        expected = np.array([16, 25, 36])
        return self.assertTrue(array_equal(result, expected))
    
    def test_pow_variable_forward2(self):
        x = np.array([1, 2, 3])
        result = (x**3).data
        expected = np.array([1, 8, 27])
        return self.assertTrue(array_equal(result, expected))

    def test_pow_variable_backward(self):
        x = Variable(np.random.randn(3, 3))
        f = lambda x: x**2    
        self.assertTrue(gradient_check(f, x))

    def test_pow_variable_backward2(self):
        x = Variable(np.random.randn(5, 5))
        f = lambda x: x**3    
        self.assertTrue(gradient_check(f, x))
    



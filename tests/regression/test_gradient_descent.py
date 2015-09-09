import unittest
from unittest import TestCase
import numpy
from ...regression.gradient_descent import (
    gradient_descent,
    stochastic_gradient_descent,
    adagrad)
from ...util import loss_functions
from scipy.sparse import csc_matrix


class GradientDescentTest(TestCase):

    def test_ndarray(self):
        x = numpy.zeros([10, 2])
        x[:, 0] = [1] * len(x)
        x[:, 1] = range(len(x))

        y = numpy.zeros([10, 1])
        y[:, 0] = range(len(x))
        y *= 2

        theta = numpy.random.rand(2, 1)
        coef = gradient_descent(loss_functions.linear,
                                loss_functions.dlinear,
                                x,
                                y,
                                theta,
                                500)
        sse = sum(sum(map(abs, y - x.dot(coef))))
        self.assertLess(sse, 1)

    def test_multiple_regression_ndarray(self):
        x = numpy.zeros([10, 3])
        x[:, 0] = [1] * len(x)
        x[:, 1] = range(len(x))
        backwards = range(len(x))
        backwards.reverse()
        x[:, 2] = backwards

        y = numpy.zeros([10, 2])
        y[:, 0] = range(len(x))
        y[:, 1] = y[:, 0] * 2

        theta = numpy.random.rand(3, 2)
        coef = gradient_descent(loss_functions.linear,
                                loss_functions.dlinear,
                                x,
                                y)

        sse = sum(sum(map(abs, y - x.dot(coef))))
        self.assertLess(sse, 1)


class StochasticGradientDescentTest(TestCase):

    def setUp(self):
        self.x = numpy.zeros([10, 3])
        self.x[:, 0] = [1] * len(self.x)
        self.x[:, 1] = range(len(self.x))
        backwards = range(len(self.x))
        backwards.reverse()
        self.x[:, 2] = backwards

        self.y = numpy.zeros([10, 2])
        self.y[:, 0] = range(len(self.x))
        self.y[:, 1] = self.y[:, 0] * 2


    def test_multiple_regression_batch(self):
        coef = stochastic_gradient_descent(loss_functions.linear,
                                           loss_functions.dlinear,
                                           self.x,
                                           self.y,
                                           iterations=500,
                                           batch_size=5)

        sse = sum(sum(map(abs, self.y - self.x.dot(coef))))
        self.assertLess(sse, 1)

    def test_proportion_batch(self):
        coef = stochastic_gradient_descent(loss_functions.linear,
                                           loss_functions.dlinear,
                                           self.x,
                                           self.y,
                                           iterations=500,
                                           batch_size=.5)

        sse = sum(sum(map(abs, self.y - self.x.dot(coef))))
        self.assertLess(sse, 1)

    def test_uneven_size_batch(self):
        coef = stochastic_gradient_descent(loss_functions.linear,
                                           loss_functions.dlinear,
                                           self.x,
                                           self.y,
                                           iterations=1000,
                                           batch_size=3)

        sse = sum(sum(map(abs, self.y - self.x.dot(coef))))
        self.assertLess(sse, 1)


class AdagradTest(TestCase):
    def test_exponential_smoothing(self):
        x = numpy.zeros([10, 2])
        x[:, 0] = [1] * len(x)
        x[:, 1] = range(len(x))

        y = numpy.zeros([10, 1])
        y[:, 0] = range(len(x))
        y *= 2

        theta = numpy.random.rand(2, 1)
        coef = adagrad(loss_functions.linear,
                       lambda x: x,
                       x,
                       y,
                       theta,
                       1000,
                       smoothing=.5)

        sse = sum(sum(map(abs, y - x.dot(coef))))
        self.assertLess(sse, 1)

    def test(self):
        x = numpy.zeros([10, 2])
        x[:, 0] = [1] * len(x)
        x[:, 1] = range(len(x))

        y = numpy.zeros([10, 1])
        y[:, 0] = range(len(x))
        y *= 2

        theta = numpy.random.rand(2, 1)
        coef = adagrad(loss_functions.linear,
                       lambda x: x,
                       x,
                       y,
                       theta,
                       1000)

        sse = sum(sum(map(abs, y - x.dot(coef))))
        self.assertLess(sse, 1)


if __name__ == '__main__':
    unittest.main()

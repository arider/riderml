import numpy
import unittest
from ...regression.SGD_regressor import SGD_regressor
from ...util.loss_functions import linear, logistic_binary, logistic, dlogistic
from unittest import TestCase
from ...util.preprocessing import as_matrix


class SGDRegressorTest(TestCase):

    def test_ndarray(self):
        x = numpy.zeros([10, 1])
        x[:, 0] = range(len(x))

        y = numpy.zeros([10, 1])
        y[:, 0] = range(len(x))
        y *= 2

        model = SGD_regressor()
        model.fit(x, y, 500)
        predicted = as_matrix(model.predict(x))
        
        mae = sum(map(abs, y - predicted))[0] / len(predicted)
        print mae
        self.assertLess(mae, 1)


    def test_learns_intercept(self):
        data = numpy.array(range(0, 10)).reshape(10, 1)
        y = numpy.array([i + 10 for i in range(0, 10)]).reshape(10, 1)

        model = SGD_regressor(function=linear)
        model.fit(data, y, 1000)
        predicted = as_matrix(model.predict(data))
        # check it it learned the y-intercept
        self.assertLess(predicted[0] - 50, 5)


if __name__ == '__main__':
    unittest.main()

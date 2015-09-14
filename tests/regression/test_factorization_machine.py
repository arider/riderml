import numpy
import unittest
from ...regression.factorization_machine import factorization_machine
from unittest import TestCase
from scipy.sparse import csc_matrix, hstack, lil_matrix
from ...util.preprocessing import row_indicators, sparse_encoding
import random


class FactorizationMachineTest(TestCase):

    def test_regression(self):
        data = numpy.array([[1, 6],
                            [2, 3],
                            [3, 0]])

        factor_x = hstack([row_indicators(data, data.shape[1]),
                           sparse_encoding(data)], format='csc')

        factor_y = csc_matrix(numpy.array([10, -30, 20, -20, 30, -10]))

        model = factorization_machine()
        model.fit(factor_x, factor_y, iterations=100)
        self.assertLess((factor_y - model.predict(factor_x)).sum(), .1)


if __name__ == '__main__':
    unittest.main()

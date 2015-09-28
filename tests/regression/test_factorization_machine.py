import numpy
import unittest
from ...regression.factorization_machine import factorization_machine
from unittest import TestCase
from scipy.sparse import hstack
from ...util.preprocessing import row_indicators, sparse_encoding


class FactorizationMachineTest(TestCase):

    def test_simple_regression(self):
        """
        It should be able to get very close....
        TODO: write some real tests
        """
        data = numpy.array([[1, 6],
                            [2, 3],
                            [3, 0]])
        factor_y = numpy.array([10, -30, 20, -20, 30, -10])

        factor_x = hstack([row_indicators(data.shape[0], data.shape[1]),
                           sparse_encoding(data)], format='csc')

        model = factorization_machine(n_factors=10, debug=False)
        model.fit(factor_x, factor_y, iterations=100)
        predicted = model.predict(factor_x)[0]
        mae = sum(map(abs, factor_y - predicted))
        self.assertLess(mae, 2.)

        factor_y = numpy.array([10, 0, 20, 0, 30, 0])
        model = factorization_machine(n_factors=10, debug=False)
        model.fit(factor_x, factor_y, iterations=100)
        predicted = model.predict(factor_x)[0]
        mae = sum(map(abs, factor_y - predicted))
        self.assertLess(mae, 2.)

if __name__ == '__main__':
    unittest.main()

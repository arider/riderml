import numpy
from ...regression.factorization_machine import factorization_machine
import random
from scipy.sparse import csc_matrix, hstack, lil_matrix
import unittest
from unittest import TestCase
from ...util.preprocessing import row_indicators, sparse_encoding
from sklearn.datasets import load_iris


class FactorizationMachineTest(TestCase):

    def test_simple_regression(self):
        data = numpy.array([[1, 6],
                            [2, 3],
                            [3, 0]])
        factor_y = numpy.array([10, -30, 20, -20, 30, -10])

        factor_x = hstack([row_indicators(data.shape[0], data.shape[1]),
                           sparse_encoding(data)], format='csc')

        model = factorization_machine(n_factors=10, debug=False)
        model.fit(factor_x, factor_y, iterations=100)
        predicted = model.predict(factor_x)
        mae = sum(map(abs, factor_y - predicted))
        self.assertLess(mae, 1.)

        factor_y = numpy.array([10, 0, 20, 0, 30, 0])
        model = factorization_machine(n_factors=10, debug=False)
        model.fit(factor_x, factor_y, iterations=100)
        predicted = model.predict(factor_x)
        mae = sum(map(abs, factor_y - predicted))
        self.assertLess(mae, 1.)

    def test_fill_noised_iris_matrix(self):
        """
        Test that the model fit improves on a real data set
        """
        x = load_iris().data

        # get the true values for all elements
        factor_y = []
        for ri in range(x.shape[0]):
            for ci in range(x.shape[1]):
                factor_y.append(x[ri, ci])
        factor_y = numpy.array(factor_y)

        # randomly assign 1/4th of them to zero
        p = .25
        for ri in range(len(x)):
            for ci in range(x.shape[1]):
                if random.random() < p:
                    x[ri, ci] = 0.

        # make a sparse matrix (no relational entries though)
        factor_x = hstack([row_indicators(x.shape[0], x.shape[1]),
                           sparse_encoding(x)], format='csc')

        # split into test and train set
        cv = int(factor_x.shape[0] * .8)
        inds = numpy.random.permutation(factor_x.shape[0])
        train_x = factor_x[inds[:cv]]
        train_y = factor_y[inds[:cv]]
        test_x = factor_x[inds[cv:]]
        test_y = factor_y[inds[cv:]]

        model = factorization_machine(debug=True)
        model.fit(train_x, train_y, iterations=1)
        predicted = model.predict(test_x)
        first_mae = sum(map(abs, test_y - predicted))

        model.fit(train_x, train_y, iterations=20)
        predicted = model.predict(test_x)
        second_mae = sum(map(abs, test_y - predicted))
        self.assertLess(second_mae, first_mae)

    def test_update_learning_rates(self):
        model = factorization_machine(n_factors=10, debug=False)
        weights, delta = model.update_learning_rates(
            numpy.array([[1, 1, -1, -1, 0, 0]]),
            numpy.array([[1, -1, 1, -1, 1, -1]]),
            numpy.array([[1., 1., 1., 1., 1., 1.]]),
            numpy.array([[0., 0., 0., 0., 0., 0.]]))

        self.assertItemsEqual(weights[0], [-1.2, 0., 0., 1.2, 0., 0.])
        self.assertItemsEqual(delta[0], [1.2, 0.5, 0.5, 1.2, 1., 1.])

    def test_update_features_step(self):
        pass

    def test_update_factors_step(self):
        pass

    def test_regularization_step(self):
        pass

    def test_predict(self):
        pass


if __name__ == '__main__':
    unittest.main()

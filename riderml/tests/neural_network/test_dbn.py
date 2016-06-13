import numpy
import unittest
from ...neural_network import nn
from ...neural_network.dbn import dbn
from ...util.evaluation import MAE
from ...util.linear_functions import (
    linear,
    dlinear,
    tanh,
    dtanh)
from unittest import TestCase
from sklearn import datasets


class DBNTest(TestCase):
    def setUp(self):
        self.x = datasets.load_iris().data
        self.y = datasets.load_iris().target
        # test without pretraining
        self.model = dbn([nn.layer(4, linear, dlinear),
                          nn.layer(5, tanh, dtanh),
                          nn.layer(1, linear, dlinear, bias=False)], False)

    def test_init_weights(self):
        self.assertFalse(hasattr(self.model.layers[0], 'weights'))
        self.assertTrue(self.model.layers[1].weights is not None)
        self.assertTrue(self.model.layers[2].weights is not None)

    def test_propagate_forward(self):
        self.model.fit(self.x, self.y, 10)
        original = self.model.layers[-1].visible
        self.model.propagate_forward(self.x[0])
        encoded = self.model.layers[-1].visible
        all_equal = True
        for index, o in enumerate(original):
            if encoded[index] != o:
                all_equal = False
        self.assertFalse(all_equal)

    def test_fit_improves(self):
        inds = numpy.random.permutation(len(self.x))
        cv = 100

        self.model.fit(self.x[inds[:cv]],
                       self.y[inds[:cv]], iterations=1)
        predicted = self.model.predict(self.x[inds[cv:]])
        predicted = [p[0] for p in predicted]
        before_error = MAE(self.y[inds[cv:]], predicted)

        self.model.fit(self.x[inds[:cv]], self.y[inds[:cv]], iterations=100)
        predicted = self.model.predict(self.x[inds[cv:]])
        predicted = [p[0] for p in predicted]
        after_error = MAE(self.y[inds[cv:]], predicted)

        self.assertLess(after_error, before_error)

    def tearDown(self):
        self.model = dbn([nn.layer(4, linear, dlinear),
                          nn.layer(5, tanh, dtanh),
                          nn.layer(1, linear, dlinear, bias=False)])


if __name__ == '__main__':
    unittest.main()

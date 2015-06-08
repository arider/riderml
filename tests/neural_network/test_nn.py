import numpy
import unittest
from ...util.linear_functions import tanh, dtanh
from ...neural_network.nn import (
    layer,
    propagate_forward,
    propagate_backward_irpropm,
    propagate_backward)
from unittest import TestCase
from functools import partial


class PropagateForwardTest(TestCase):

    def test_linear(self):
        layers = []
        layers.append(layer(2))
        layers.append(layer(3))
        layers[0].visible = numpy.array([1, 2])
        layers[1].init_weights(len(layers[0].visible))
        layers[1].weights = numpy.ones([len(layers[0].visible),
                                        len(layers[1].visible)])
        propagate_forward(layers[1], layers[0].visible)
        self.assertItemsEqual(layers[1].visible, [1, 3, 3, 3])

    def test_tanh(self):
        layers = []
        layers.append(layer(2, tanh, dtanh))
        layers.append(layer(3, tanh, dtanh))
        layers[0].visible = numpy.array([1, 2])
        layers[1].init_weights(len(layers[0].visible))
        layers[1].weights = numpy.ones([len(layers[0].visible),
                                        len(layers[1].visible)])

        propagate_forward(layers[1], layers[0].visible)

        self.assertAlmostEqual(layers[1].visible[0], 1., 3)
        self.assertAlmostEqual(layers[1].visible[1], 0.99505475, 3)
        self.assertAlmostEqual(layers[1].visible[2], 0.99505475, 3)
        self.assertAlmostEqual(layers[1].visible[3], 0.99505475, 3)


class PropagateBackwardTest(TestCase):
    def test(self):
        layers = []
        layers.append(layer(2))
        layers.append(layer(3, backward_function=propagate_backward))

        layers[0].visible = numpy.array([1, 2])
        layers[1].init_weights(len(layers[0].visible))
        layers[1].weights = numpy.ones([len(layers[0].visible),
                                        len(layers[1].visible)])

        target = numpy.array([1, 2, 2, 2])
        layers[1].propagate_backward(layers[0], target)
        self.assertEqual(layers[1].weights[0, 0], 1.)
        self.assertEqual(layers[1].weights[0, 1], 1.1)
        self.assertEqual(layers[1].weights[0, 2], 1.1)
        self.assertEqual(layers[1].weights[0, 3], 1.1)

        self.assertEqual(layers[1].weights[1, 0], 1.)
        self.assertEqual(layers[1].weights[1, 1], 1.2)
        self.assertEqual(layers[1].weights[1, 2], 1.2)
        self.assertEqual(layers[1].weights[1, 3], 1.2)


class PropagateBackwardIrpropmTest(TestCase):
    def test_update_weights(self):
        partial_bp = partial(propagate_backward_irpropm, batch_size=2)

        layers = []
        layers.append(layer(2))
        layers.append(layer(3, backward_function=partial_bp))

        layers[0].visible = numpy.array([1, 2])
        layers[1].init_weights(len(layers[0].visible))
        layers[1].visible = numpy.array([1., 2, 2, 2])

        layers[1].weights = numpy.ones([len(layers[0].visible),
                                        len(layers[1].visible)])

        self.assertTrue(
            numpy.all(layers[1].weights
                      == numpy.ones(layers[1].weights.shape)))
        target = numpy.array([1, 3, 3, 3])
        # bath size is 2, so we need three calls since the first
        # one initializes the layer
        layers[1].propagate_backward(layers[0], target)

        self.assertTrue(
            numpy.all(layers[1].weights == numpy.array(
                [[1, 1.1, 1.1, 1.1],
                 [1, 1.1, 1.1, 1.1]])))

    def test_update_step_size(self):
        partial_bp = partial(propagate_backward_irpropm, batch_size=1)

        layers = []
        layers.append(layer(2))
        layers.append(layer(3, backward_function=partial_bp))

        layers[0].visible = numpy.array([1, 2])
        layers[1].init_weights(len(layers[0].visible))
        layers[1].visible = numpy.array([1., 2, 2, 2])

        layers[1].weights = numpy.ones([len(layers[0].visible),
                                        len(layers[1].visible)])

        self.assertTrue(
            numpy.all(layers[1].weights
                      == numpy.ones(layers[1].weights.shape)))
        target = numpy.array([1, 3, 3, 3])
        # we need three calls here since the sign of the gradient
        # won't be established until the second call.
        layers[1].propagate_backward(layers[0], target)
        layers[1].propagate_backward(layers[0], target)
        layers[1].propagate_backward(layers[0], target)

        for ri in [0, 1]:
            for ci in [1, 2, 3]:
                self.assertAlmostEqual(layers[1].step_size[ri, ci], .14, 2)


if __name__ == '__main__':
    unittest.main()

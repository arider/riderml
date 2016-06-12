import numpy
import unittest
from ...neural_network.autoencoder import autoencoder
from ...util.evaluation import MAE
from unittest import TestCase
from mock import patch
from sklearn import datasets


class AutoencoderTest(TestCase):
    def setUp(self):
        self.x = datasets.load_iris().data

    def test_propagate_forward(self):
        model = autoencoder(5)
        model.fit(self.x, 10)
        original = model.layers[-1].visible
        model.propagate_forward(self.x[0])
        encoded = model.layers[-1].visible
        all_equal = True
        for index, o in enumerate(original):
            if encoded[index] != o:
                all_equal = False
        self.assertFalse(all_equal)

    def test_fit(self):
        # test initializes weights
        model = autoencoder(5)
        model.fit(self.x, 10)
        for layer in model.layers:
            self.assertIsNotNone(layer)

        # test noise was called
        with patch.object(autoencoder, 'noise') as mock_method:
            model.fit(self.x, 1)
            self.assertTrue(mock_method.called)

    def test_noise(self):
        model = autoencoder(5)
        test = model.noise(self.x)
        # check that there are at least 10 zeros.... should pass
        count = 0
        for row in test:
            count += sum([1 for e in row if e == 0])

        self.assertGreaterEqual(count, 10)

    def test_fit_improves(self):
        model = autoencoder(5)
        model.fit(self.x, iterations=1, noise=.1)
        predicted = numpy.array(model.predict(self.x))
        before_error = MAE(self.x, predicted)

        model.fit(self.x, iterations=100, noise=.1)
        predicted = numpy.array(model.predict(self.x))
        after_error = MAE(self.x, predicted)

        self.assertLess(after_error, before_error)


if __name__ == '__main__':
    unittest.main()

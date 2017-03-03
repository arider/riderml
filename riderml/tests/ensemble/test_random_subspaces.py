import numpy
import unittest
from unittest import TestCase
from ...regression.SGD_regressor import SGD_regressor
from ...ensemble.random_subspaces import random_subspaces
from ...util.preprocessing import as_matrix


class RandomSubspacesTest(TestCase):
    def setUp(self):
            self.x = numpy.zeros([10, 2])
            self.x[:, 0] = numpy.array(range(0, 10))
            self.x[:, 1] = numpy.array(range(0, 10)) + 5

            self.y = numpy.zeros([10, 2])
            self.y[:, 0] = numpy.array([i for i in range(0, 10)])
            self.y[:, 1] = numpy.array([i + 10 for i in range(0, 10)])

    def test_learn_one_output(self):
        model = random_subspaces(
            model_class=SGD_regressor,
            init_args={},
            fit_args={'iterations': 1000},
            n_models=3,
            subspace_size=0,
            response_matching=False)

        model.fit(self.x, numpy.atleast_2d(self.y[:, 0]).T)
        self.assertEqual(len(model.model_features), 3)
        for subspace in model.model_features:
            self.assertGreaterEqual(len(subspace), 1)

    def test_learn_multi_output_not_response_matching(self):
        model = random_subspaces(
            SGD_regressor,
            {},
            {'iterations': 1000},
            3,
            0,
            False)
        model.fit(self.x, self.y)
        predicted, count = model.predict(self.x)
        self.assertEqual(predicted.shape[1], self.x.shape[1])

        # when response_matching is False, each predictor predicts
        # every output... therefore, the counts should be equal to
        # the number of subspaces.
        self.assertItemsEqual(count, [3., 3.])

    def test_response_matching(self):
        model = random_subspaces(
            SGD_regressor,
            {},
            {'iterations': 1000},
            20,
            1,
            True)
        model.fit(self.x, self.x)
        predicted, count = model.predict(self.x)

        # this one could fail if every model in the ensemble has
        # the same set of features
#        print model.model_features
        self.assertNotEqual(1, len(set(count)))

    def test_fit_with_model_indices(self):
        model = random_subspaces(
            SGD_regressor,
            {},
            {'iterations': 1000},
            10,
            1,
            False)
        model.fit(self.x, self.y)
        model.models = [None] * 10
        model.fit(self.x, self.y, [1, 3, 5])
        self.assertIsNone(model.models[0])
        self.assertIsNone(model.models[2])
        self.assertIsNone(model.models[4])
        self.assertIsNotNone(model.models[1])
        self.assertIsNotNone(model.models[3])
        self.assertIsNotNone(model.models[5])

    def test_single_dimension_dataset(self):
        model = random_subspaces(
            model_class=SGD_regressor,
            init_args={},
            fit_args={'iterations': 1000},
            n_models=3,
            subspace_size=0,
            response_matching=False)

        model.fit(as_matrix(numpy.atleast_2d(self.x[:, 0]).T), numpy.atleast_2d(self.y[:, 0]).T)

    def test_predict_components(self):
        model = random_subspaces(
            model_class=SGD_regressor,
            init_args={},
            fit_args={'iterations': 1000},
            n_models=3,
            subspace_size=0,
            response_matching=False)

        model.fit(self.x, numpy.atleast_2d(self.y[:, 0]).T)
        results, counts = model.predict_components(self.x)
        self.assertEqual(len(results), model.n_models)
        self.assertEqual(len(results[0]), self.x.shape[0])

    def test_predict_components_response_matching(self):
        model = random_subspaces(
            model_class=SGD_regressor,
            init_args={},
            fit_args={'iterations': 1000},
            n_models=3,
            subspace_size=0,
            response_matching=False)

        model.fit(self.x, numpy.atleast_2d(self.y[:, 0]).T)
        results, counts = model.predict_components(self.x)
        self.assertEqual(len(results), model.n_models)
        self.assertEqual(len(results[0]), self.x.shape[0])


if __name__ == '__main__':
    unittest.main()

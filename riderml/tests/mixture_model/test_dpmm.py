import unittest
from unittest import TestCase
from ...mixture_model.dpmm import (
    multinomial,
    dpmm,
    gaussian)
import numpy


class GaussianTest(TestCase):
    def setUp(self):
        self.dist = gaussian(2)
        self.dist.add_i(numpy.array([1, 2]))

    def test_update_params(self):
        self.dist.data.append(numpy.array([200, 300]))
        old_sigma = self.dist.sigma
        old_mu = self.dist.mu
        self.dist.update_params()
        self.assertNotEqual(tuple(old_mu), tuple(self.dist.mu))
        self.assertNotEqual(old_sigma.shape, self.dist.sigma.shape)

    def test_posterior(self):
        # TODO: make this a proper test
        old_posterior = self.dist.posterior(numpy.array([1, 2]))
        self.dist.add_i(numpy.array([2, 1]))
        new_posterior = self.dist.posterior(numpy.array([1, 2]))
        self.assertNotEqual(old_posterior, new_posterior)

    def test_rem_i(self):
        self.dist.rem_i(numpy.array([1, 2]))
        self.assertEqual(0, len(self.dist.data))

    def test_add_i(self):
        self.dist.add_i(numpy.array([2, 1]))
        self.assertTrue((2, 1) in self.dist.data)


class MultinomialTest(TestCase):

    def setUp(self):
        self.mn = multinomial(3)
        self.mn.add_i(numpy.array([1, 2, 3]))

    def test_rem_i(self):
        self.mn.rem_i(numpy.array([3, 2, 1]))
        self.assertItemsEqual(self.mn.phi, [-2, 0, 2])

    def test_add_i(self):
        self.mn.add_i(numpy.array([3, 2, 1]))
        self.assertItemsEqual(self.mn.phi, [4, 4, 4])

    def test_posterior(self):
        self.assertAlmostEqual(-4.17,
                               self.mn.posterior(numpy.array([3, 2, 1])), 2)

    def tearDown(self):
        self.mn = multinomial(3)


class DpmmTest(TestCase):
    """
    Test dpmm with multinomial base distributions.
    """
    def setUp(self):
        self.data = []
        for i in range(50):
            self.data.append(numpy.random.multinomial(
                20, [.5, .2, .1, .05, .03, .02], size=1)[0])
            self.data.append(numpy.random.multinomial(
                20, [.02, .03, .05, .1, .2, .5], size=1)[0])
            self.data.append(numpy.random.multinomial(
                20, [.1, .1, .3, .3, .1, .1], size=1)[0])
        self.data = numpy.array(self.data)

    def test_multinomial_ll(self):
        model = dpmm(self.data, .1, .000001, 1)
        model.components[0].phi = numpy.array([1, 1, 1, 0, 0, 0])
        p = model.components[0].posterior(numpy.array([1, 1, 1, 0, 0, 0]))
        self.assertAlmostEqual(-1.50408039677, p, 3)
        p = model.components[0].posterior(numpy.array([0, 0, 0, 1, 1, 1]))
        self.assertAlmostEqual(-49.858364949651175, p, 2)
        p = model.components[0].posterior(numpy.array([1, 0, 0, 0, 0, 1]))
        self.assertAlmostEqual(-17.622, p, 1)

    def test_pc_x(self):
        # create a model with the given parameters
        self.data[0] = numpy.array([1, 1, 1, 1, 0, 0])
        model = dpmm(self.data, .1, .000001, 1)
        # needs a new_phi even though it won't be used in the test
        comp = multinomial(6)
        comp.add_i(numpy.array([0, 0, 1, 1, 1, 1]))
        model.new_components.append(comp)

        # this component should be the most likely
        comp = multinomial(6)
        for i in range(100):
            comp.add_i(numpy.array([1, 1, 1, 1, 0, 0]))
        model.components[0] = comp

        comp = multinomial(6)
        comp.add_i(numpy.array([0, 0, 1, 1, 1, 1]))
        model.components.append(comp)

        model.c = [0] * self.data.shape[0]
        results = model.pc_x(0)

        self.assertAlmostEqual(1.0, sum(results), 2)
        self.assertEqual(numpy.array(results).argmax(), 0)

    def test_add_rand_phi(self):
        model = dpmm(self.data, .1, .000001, 1)
        original_len = len(model.new_components)
        model.add_rand_phi()

        self.assertEqual(len(model.new_components), original_len + 1)
        # there should be at least 20 in the new component since it randomly
        # selects between 1 and 10 observations and each observation has at
        # least 20 draws in it
        self.assertGreaterEqual(model.new_components[-1].phi.sum(), 20)

    def test_rem_i(self):
        self.data[0] = numpy.array([1, 1, 0, 1, 0, 0])
        self.data[1] = numpy.array([1, 1, 0, 1, 0, 0])
        model = dpmm(self.data, .1, .000001, 1)
        model.rem_i(0)
        self.assertItemsEqual(model.components[0].phi, self.data[0])

    def test_add_i(self):
        self.data[0] = numpy.array([1, 1, 0, 1, 0, 0])
        self.data[1] = numpy.array([1, 1, 0, 1, 0, 0])
        model = dpmm(self.data, .1, .000001, 1)
        model.add_i(0, 0)
        self.assertItemsEqual(self.data[0] * 2,
                              model.components[0].phi - self.data[0])

    def test_get_labels(self):
        # large alpha to encourage many clusters...
        model = dpmm(self.data, 1., .000001, 1)
        model.fit(10)
        labels = model.get_labels(self.data)
        self.assertEqual(len(labels), self.data.shape[0])
        em_labels = model.get_labels(self.data, True)

        all_equal = True
        for i in range(len(labels)):
            if labels[i] != em_labels[i]:
                all_equal = False
                break
        self.assertFalse(all_equal)

    def test_update_data(self):
        model = dpmm(self.data, 1., .000001, 1)
        inds = numpy.random.permutation(range(self.data.shape[0]))
        sample = self.data[inds[:10]]

        model.update_data(sample)

        # updates data
        self.assertEqual(model.N, sample.shape[0])
        self.assertEqual(sample.shape[0], model.data.shape[0])
        # removes empty clusters
        for component in model.components:
            self.assertGreater(component.n, 0)

    def test_identifies_three_multinomials(self):
        model = dpmm(self.data, .1, .000001, 1)
        model.fit(20)
        counts = sorted([c.n for c in model.components])
        counts.reverse()
        # most of the instances should be in the three largest components
        self.assertGreaterEqual(sum(counts[:3]), 100)

    def test_fit_bagging(self):
        model = dpmm(self.data, .1, .000001, 1)
        model.fit_bagging(self.data, 20, .8)
        counts = sorted([c.n for c in model.components])
        counts.reverse()
        # most of the instances should be in the three largest components
        self.assertGreaterEqual(sum(counts[:3]), 100)


if __name__ == '__main__':
    unittest.main()

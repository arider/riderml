from ...util.linear_functions import (
    hellinger_distance,
    normalized_hellinger_distance,
    hellinger_distance_prebin)
import unittest
from unittest import TestCase
import numpy


class HellingerDistanceTest(TestCase):
    def test_max(self):
        a = [2, 2, 2, 2, 2, 2, 2, 2, 9, 9]
        b = [9, 9, 9, 9, 9, 9, 9, 9, 2, 2]
        self.assertAlmostEqual(.707, hellinger_distance(a, b, 2), 3)

    def test_min(self):
        a = [9, 9, 9, 9, 9, 9, 9, 9, 2, 2]
        b = [9, 9, 9, 9, 9, 9, 9, 9, 2, 2]
        self.assertEqual(0, hellinger_distance(a, b, 2))


class HellingerDistancePrebinTest(TestCase):
    def test(self):
        a = numpy.zeros((20, 1))
        b = numpy.zeros((20, 1))
        a = range(1, 21)
        b = range(20, 0, -1)
        self.assertAlmostEqual(0.3111, hellinger_distance_prebin(a, b), 3)


class NormalizedHellingerDistanceTest(TestCase):
    def test_max(self):
        a = [2, 2, 2, 2, 2, 2, 2, 2, 9, 9]
        b = [9, 9, 9, 9, 9, 9, 9, 9, 2, 2]
        self.assertAlmostEqual(.5, normalized_hellinger_distance(a, b, 2), 1)

    def test_min(self):
        a = [9, 9, 9, 9, 9, 9, 9, 9, 2, 2]
        b = [9, 9, 9, 9, 9, 9, 9, 9, 2, 2]
        self.assertEqual(0, normalized_hellinger_distance(a, b, 2))

if __name__ == '__main__':
    unittest.main()

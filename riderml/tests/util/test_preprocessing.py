import numpy
import unittest
from unittest import TestCase
from ...util.preprocessing import (
    as_matrix,
    as_row,
    sparse_filtering_normalizer,
    bin_data,
    sparse_encoding,
    sparse_implicit_encoding,
    row_indicators)


class SparseFilteringNormNormalizerTest(TestCase):
    def test_list_normalize(self):
        data = [1, 2, 3]
        normalizer = sparse_filtering_normalizer(data)
        normalized = normalizer.normalize(data)

        self.assertAlmostEqual(0.26726124, normalized[0])
        self.assertAlmostEqual(0.53452248, normalized[1])
        self.assertAlmostEqual(0.80178373, normalized[2])

        denormalized = normalizer.denormalize(normalized)

        self.assertAlmostEqual(1, denormalized[0])
        self.assertAlmostEqual(2, denormalized[1])
        self.assertAlmostEqual(3, denormalized[2])

    def test_matrix(self):
        data = numpy.zeros([3, 2])
        data[:, 0] = [1, 2, 3]
        data[:, 1] = [10, 20, 30]
        normalizer = sparse_filtering_normalizer(data)
        normalized = normalizer.normalize(data)
        denormalized = normalizer.denormalize(normalized)

        for row in normalized:
            for el in row:
                self.assertAlmostEqual(0.70710678, el, 2)

        for ri, row in enumerate(denormalized):
            for ci, el in enumerate(row):
                self.assertAlmostEqual(el, data[ri, ci])


class AsMatrixTest(TestCase):
    def test_1d(self):
        data = [1, 2, 3]
        copy = as_matrix(data)
        self.assertEqual(copy.shape[0], 3)
        self.assertEqual(copy.shape[1], 1)

    def test_2d(self):
        data = [[1, 2, 3], [4, 5, 6]]
        copy = as_matrix(data)
        self.assertEqual(copy.shape[0], 2)
        self.assertEqual(copy.shape[1], 3)

    def test_is_array(self):
        data = numpy.array([[1, 2, 3], [4, 5, 6]])
        copy = as_matrix(data)
        self.assertEqual(copy.shape[0], 2)
        self.assertEqual(copy.shape[1], 3)


class AsRowTest(TestCase):
    def test_1d(self):
        data = [1, 2, 3]
        copy = as_row(data)
        self.assertEqual(copy.shape[0], 1)
        self.assertEqual(copy.shape[1], 3)

    def test_scalar(self):
        data = 2
        copy = as_row(data)
        self.assertEqual(copy.shape[0], 1)
        self.assertEqual(copy.shape[1], 1)

    def test_2d(self):
        data = [[1, 2, 3], [4, 5, 6]]
        try:
            as_row(data)
            self.assertTrue(False)
        except:
            self.assertTrue(True)


class BinDataTest(TestCase):
    def test_upper_edge(self):
        x = range(10)
        bins = bin_data(x, 5)
        self.assertItemsEqual(bins, [2, 2, 2, 2, 2])

    def test_no_binning(self):
        x = range(10)
        bins = bin_data(x, 10)
        self.assertItemsEqual(bins, [1] * 10)

    def test_uniform_input(self):
        x = [10] * 10
        bins = bin_data(x, 5)
        self.assertEqual(bins[0], 10)
        self.assertEqual(bins[1:].sum(), 0)


class RowIndicatorsTest(TestCase):
    def test(self):
        solution = numpy.array([[1.,  0.,  0.],
                                [1.,  0.,  0.],
                                [0.,  1.,  0.],
                                [0.,  1.,  0.],
                                [0.,  0.,  1.],
                                [0.,  0.,  1.]])
        indicators = numpy.array(row_indicators(3, 2).todense())

        for i in range(len(indicators)):
            self.assertItemsEqual(solution[i], indicators[i])

    def test_again(self):
        solution = numpy.array([[1, 0],
                                [1, 0],
                                [1, 0],
                                [0, 1],
                                [0, 1],
                                [0, 1]])

        out = numpy.array(row_indicators(2, 3).todense())

        for i in range(len(out)):
            self.assertItemsEqual(solution[i], out[i])


class SparseEncodingIndicatorsTest(TestCase):
    def test(self):
        data = numpy.array([[2, 4],
                            [4, 6],
                            [6, 8]])
        solution = numpy.array([[1.,  0.],
                                [0.,  1.],
                                [1.,  0.],
                                [0.,  1.],
                                [1.,  0.],
                                [0.,  1.]])
        indicators = numpy.array(sparse_encoding(data, 2).todense())

        for i in range(len(indicators)):
            self.assertItemsEqual(solution[i], indicators[i])

    def test_not_binary(self):
        data = numpy.array([[1, 2, 3],
                            [4, 5, 6]])
        solution = numpy.array([[1., 0., 0.],
                                [0., 2., 0.],
                                [0., 0., 3.],
                                [4., 0., 0.],
                                [0., 5., 0.],
                                [0., 0., 6.]])

        out = numpy.array(sparse_encoding(data, False).todense())

        for i in range(len(out)):
            self.assertItemsEqual(solution[i], out[i])


class SparseImplicitEncodingTest(TestCase):
    def test_self_interactions_binary(self):
        data = numpy.array([[1, 1],
                            [1, 0],
                            [1, 1]])

        solution = numpy.array([[.5, .5],
                                [.5, .5],
                                [1., 0.],
                                [1., 0.],
                                [.5, .5],
                                [.5, .5]])

        out = numpy.array(sparse_implicit_encoding(data).todense())
        for i in range(len(out)):
            self.assertItemsEqual(solution[i], out[i])

    def test_self_interactions_not_binary(self):
        data = numpy.array([[2, 4],
                            [4, 0],
                            [6, 8]])
        out = numpy.array(sparse_implicit_encoding(data, False).todense())
        self.assertAlmostEqual(out[0, 0], .333, 2)
        self.assertAlmostEqual(out[0, 1], .666, 2)
        self.assertAlmostEqual(out[4, 0], .428, 2)


if __name__ == '__main__':
    unittest.main()

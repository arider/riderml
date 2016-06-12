import numpy
from ...mixture_model.lda import lda
import unittest
from unittest import TestCase


class ldaTest(TestCase):
    def setUp(self):
        self.data = numpy.array([[4, 3, 2, 1],
                                 [3, 2, 1, 0],
                                 [2, 2, 0, 0],
                                 [0, 0, 2, 2],
                                 [0, 1, 2, 3],
                                 [1, 2, 3, 4]])

        self.model = lda(self.data, 2, 4)

    def test_word_counts_to_indices(self):
        self.assertItemsEqual([0, 0, 1, 1],
                              self.model.word_counts_to_indices(self.data)[2])


    def test_sample_topics(self):
        # TODO: add a real test
        d = 0
        word_index = 0
        self.assertGreaterEqual(self.model.sample_topics(0, 0)[0], 0.)


    def test_fit(self):
        d = 0
        word_index = 0
        p_old = self.model.sample_topics(0, 0)
        self.model.fit(100)
        p_new = self.model.sample_topics(0, 0)
        self.assertNotEqual(p_old[0], p_new[0])
        self.assertNotEqual(p_old[1], p_new[1])

    def test_rem_dwt(self):
        self.model.add_dwt(0, 0, 0, 0)
        original_dt = self.model.dt[0, 0]
        original_tw = self.model.tw[0, 0]
        original_n_t = self.model.n_t[0]
        self.model.rem_dwt(0, 0, 0)
        self.assertTrue(self.model.dt[0, 0] == original_dt - 1)
        self.assertTrue(self.model.tw[0, 0] == original_tw - 1)
        self.assertTrue(self.model.n_t[0] == original_n_t - 1)

    def test_add_dwt(self):
        original_dt = self.model.dt[0, 0]
        original_tw = self.model.tw[0, 0]
        original_n_t = self.model.n_t[0]
        self.model.add_dwt(0, 0, 0, 0)
        self.assertTrue(self.model.dt[0, 0] == original_dt + 1)
        self.assertTrue(self.model.tw[0, 0] == original_tw + 1)
        self.assertTrue(self.model.n_t[0] == original_n_t + 1)


if __name__ == '__main__':
    unittest.main()

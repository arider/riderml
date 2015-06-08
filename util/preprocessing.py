import numpy
from abc import ABCMeta, abstractmethod


class normalizer:
    """
    base class for objects that initialize with a data set from which they
    learn the necessary features to normalize and denormalize data.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def normalize(self, x):
        pass

    @abstractmethod
    def denormalize(self, x):
        pass


class sparse_filtering_normalizer(normalizer):
    def __init__(self, x):
        self.feature_norms = None
        self.row_norms = None

        data = numpy.array(x, dtype=numpy.float64)
        if len(data.shape) == 1:
            self.feature_norms = numpy.linalg.norm(data, 2)
            data = data / self.feature_norms
            self.row_norms = numpy.ones(len(data))
        else:
            self.feature_norms = numpy.linalg.norm(data, 2, 0)
            data = data / self.feature_norms
            self.row_norms = numpy.linalg.norm(data, 2, 1)
            for ri in range(len(data)):
                data[ri, :] = data[ri, :] / self.row_norms[ri]

    def normalize(self, x):
        data = numpy.array(x, dtype=numpy.float64)
        if len(data.shape) == 1:
            data = data / self.feature_norms
        else:
            data = data / self.feature_norms
            self.row_norms = numpy.linalg.norm(data, 2, 1)
            for ri in range(len(data)):
                data[ri, :] = data[ri, :] / self.row_norms[ri]
        data = map(abs, data)
        return data

    def denormalize(self, x):
        data = numpy.array(x, dtype=numpy.float64)
        if len(data.shape) == 1:
            data = data * self.row_norms
        else:
            for ri in range(len(data)):
                data[ri, :] = data[ri, :] * self.row_norms[ri]
        data = data * self.feature_norms
        return data


def as_matrix(data):
    """
    Take a list or array of (something, ) and make it (something, 1) or an
    array
    """
    a = numpy.array(data, dtype=numpy.float64)
    if len(a.shape) == 1:
        return a.reshape(len(data), 1)
    else:
        return a


def as_row(data):
    """
    Take a list or array of (something, ) and make it (1, something) or an
    array
    """
    if hasattr(data, '__iter__'):
        a = numpy.zeros([1, len(data)], dtype=numpy.float64)
    else:
        a = numpy.zeros([1, 1], dtype=numpy.float64)

    a[0, :] = data

    return a


def bin_data(x, n_bins):
    """
    Transform the input array into a smaller list of counts.
    """
    bins = numpy.zeros(n_bins)
    mn = min(x)
    mx = max(x)
    if mn == mx:
        bins[0] += len(x)
        return bins
    bin_width = (mx - mn) / float(n_bins)

    for i in x:
        b = int((i - mn) / bin_width)
        if b == n_bins:
            b -= 1
        bins[b] += 1

    return bins

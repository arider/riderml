import numpy
from preprocessing import bin_data


def elliott(x):
    return 1 / ((1 + abs(x)) ** 2)


def delliott(x):
    return (1 - abs(x)) ** 2


def tanh(x):
    """
    Sigmoid like function using tanh
    """
    return numpy.tanh(x)


def dtanh(x):
    """
    Derivative of sigmoid above
    """
    return 1.0 - x ** 2


def linear(x):
    return x


def dlinear(x):
    return [1 for e in x]


def hellinger_distance_prebin(x, y):
    p = numpy.sqrt(numpy.array(x))
    q = numpy.sqrt(numpy.array(y))
    return (1 / numpy.sqrt(2)
            * numpy.sqrt(numpy.sum((p / len(p) - q / len(q)) ** 2)))


def normalized_hellinger_distance_prebin(x, y):
    co = numpy.sqrt(2)
    return 1 - (co - hellinger_distance_prebin(x, y)) / co


def hellinger_distance(x, y, n_bins):
    """
    args:
        x - array
        y - array
        n_bins - number of bins to use
    """
    bin_x = bin_data(x, n_bins)
    bin_y = bin_data(y, n_bins)
    return hellinger_distance_prebin(bin_x, bin_y)


def normalized_hellinger_distance(x, y, n_bins):
    """
    Normalize HD for use in weighting.
    """
    co = numpy.sqrt(2)
    return 1 - (co - hellinger_distance(x, y, n_bins)) / co

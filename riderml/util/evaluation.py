import numpy


def MAE(x, y):
    return numpy.array(map(abs, x - y)).mean()


def RMSE(x, y):
    error = numpy.array(y) - numpy.array(x)
    return numpy.sqrt(numpy.dot(error, error))

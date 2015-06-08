import numpy


def linear(x, theta):
    return numpy.dot(x, theta)


def logistic(x, theta):
    return 1 / (1 + numpy.exp(-numpy.dot(x, theta)))


def logistic_binary(x, theta):
    tmp = logistic(x, theta)
    binary = [1 if e >= .5 else 0 for e in tmp]
    return numpy.array(binary)

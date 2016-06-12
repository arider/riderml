from ...neural_network.dbn import dbn
from ...neural_network import nn
import numpy
from ...util.preprocessing import as_matrix
from ...util.linear_functions import tanh, dtanh, linear, dlinear
from ...util.evaluation import MAE
from matplotlib import pyplot
from sklearn import datasets


def test_sin_linear(iterations=500, plot=True):
    x = as_matrix(numpy.linspace(0, 1, 100))
    y = numpy.sin(x * numpy.pi * 2)
    y += 5

    model = dbn([nn.layer(1, linear, dlinear),
                 nn.layer(10, tanh, dtanh),
                 nn.layer(1, linear, dlinear, bias=False)])

    # permute the data
    inds = numpy.random.permutation(range(len(x)))

    permuted_x = x[inds]
    permuted_y = y[inds]

    cv = int(len(x) * .8)
    train_x = permuted_x[:cv, :]
    test_x = permuted_x[cv:, :]
    train_y = permuted_y[:cv]
    test_y = permuted_y[cv:]

    model.fit(train_x, train_y, iterations=iterations)
    test_predicted = as_matrix(model.predict(test_x))
    mae = MAE(test_y, test_predicted)

    if plot:
        f, ax = pyplot.subplots(1, 1)
        ax.plot(range(len(x)), y, color='b',
                label='actual')
        ax.scatter(inds[cv:], test_predicted, color='r',
                   label='test_predictions')
        ax.set_title('dbn with 1, 10, 1. mae = {}'.format(mae))
        pyplot.legend()
        pyplot.show()


def multi_line_one_input(iterations=500):
    x = numpy.linspace(0, 1, 100)
    y = numpy.zeros([len(x), 2])
    y[:, 0] = numpy.sin(x * numpy.pi * 2)
    y[:, 1] = numpy.cos(x * numpy.pi * 1.5) + 1
    x = as_matrix(x)

    model = dbn([nn.layer(1, linear, dlinear),
                 nn.layer(10, tanh, dtanh),
                 nn.layer(2, linear, dlinear, bias=False)])

    # permute the data
    inds = numpy.random.permutation(range(len(x)))

    permuted_x = x[inds]
    permuted_y = y[inds]

    cv = int(len(x) * .8)
    train_x = permuted_x[:cv, :]
    test_x = permuted_x[cv:, :]
    train_y = permuted_y[:cv]
    test_y = permuted_y[cv:]

    model.fit(train_x, train_y, iterations=iterations)
    test_predicted = as_matrix(model.predict(test_x))
    mae = MAE(test_y, test_predicted)

    f, ax = pyplot.subplots(1, 1)
    style = ['-', '--']
    for i in range(y.shape[1]):
        ax.plot(range(len(x)), y[:, i], style[i], color='b',
                label='actual', )
        ax.scatter(inds[cv:], test_predicted[:, i], color='r',
                   label='test_predictions')
    ax.set_title('dbn with 1, 10, 1. mae = {}'.format(mae))
    pyplot.legend()
    pyplot.show()


def fit_iris(iterations=100):
    x = datasets.load_iris().data
    y = datasets.load_iris().target

    model = dbn([nn.layer(4, linear, dlinear),
                 nn.layer(5, tanh, dtanh),
                 nn.layer(1, linear, dlinear, bias=False)], False)
    model.fit(x, y, iterations)

    predicted = numpy.array(model.predict(x))

    f, ax = pyplot.subplots(1, 1)
    ax.plot(range(len(x)), y, color='b', label='actual')
    ax.plot(range(len(x)), predicted, color='r', label='predicted')
    ax.legend()
    ax.set_title('MAE = {}'.format(MAE(x[:, -1],
                                   predicted[:, -1])))
    pyplot.show()

if __name__ == '__main__':
    print "training sin linear"
    test_sin_linear()
    print "training two functions."
    multi_line_one_input()
    print "fitting iris data"
    fit_iris()

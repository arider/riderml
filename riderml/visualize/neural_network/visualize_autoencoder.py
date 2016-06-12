from sklearn import datasets
import numpy
from matplotlib import pyplot
from ...neural_network.autoencoder import autoencoder
from ...util.evaluation import MAE
from ...util.linear_functions import (
    tanh,
    dtanh)
from ...util.preprocessing import sparse_filtering_normalizer


if __name__ == "__main__":
    x = datasets.load_iris().data
    y = datasets.load_iris().target

    x = numpy.zeros([150, 5])
    x[:, :-1] = datasets.load_iris().data
    x[:, -1] = y

    # sparse denoising autoencoder
    model = autoencoder(5, tanh, dtanh,
                        normalizer=sparse_filtering_normalizer)
    model.fit(x, iterations=100, noise=.1)
    noised = model.noise(x, .3)
    predicted = numpy.array(model.predict(noised))

    not_sparse_model = autoencoder(5, tanh, dtanh,
                                   normalizer=None)
    model.fit(x, iterations=100, noise=.1)
    not_sparse_predicted = numpy.array(model.predict(noised))

    f, ax = pyplot.subplots(1, 1)
    ax.plot(range(len(x)), x[:, -1], color='b', label='actual')
    ax.plot(range(len(x)), predicted[:, -1], color='r', label='predicted_sdae')
    ax.plot(range(len(x)), not_sparse_predicted[:, -1],
            color='g', label='predicted_not_sparse')
    ax.legend()
    ax.set_title('MAE = {}'.format(MAE(x[:, -1],
                                   predicted[:, -1])))
    pyplot.show()

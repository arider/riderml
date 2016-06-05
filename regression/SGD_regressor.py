from ..base import model
from gradient_descent import stochastic_gradient_descent
from ..util import loss_functions
import numpy
from ..util.preprocessing import as_matrix


class SGD_regressor(model):

    def __init__(self,
                 function=loss_functions.linear,
                 dfunction=loss_functions.dlinear,
                 theta=None,
                 learning_rate=.01,
                 fit_intercept=True):

        self.theta = theta
        self.learning_rate = learning_rate
        self.function = function
        self.dfunction = dfunction
        self.fit_intercept = fit_intercept

    def fit(self, x, y, iterations=1, shuffle=True, batch_size=.2):

        if self.fit_intercept:
            tmp = as_matrix(x)
            data = numpy.ones((tmp.shape[0], tmp.shape[1] + 1))
            data[:, 1:] = tmp
        else:
            data = as_matrix(x)

        self.theta = stochastic_gradient_descent(
            self.function,
            self.dfunction,
            data,
            y,
            self.theta,
            iterations,
            self.learning_rate,
            shuffle=shuffle,
            batch_size=batch_size)

    def predict(self, data):
        if self.fit_intercept:
            tmp = as_matrix(data)
            x = numpy.ones((tmp.shape[0], tmp.shape[1] + 1))
            x[:, 1:] = tmp

            return self.function(x, self.theta)
        else:
            return self.function(as_matrix(data), self.theta)

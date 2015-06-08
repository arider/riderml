import numpy
from ..util.linear_functions import tanh, dtanh, linear, dlinear
from ..util.preprocessing import sparse_filtering_normalizer, as_matrix
import nn


class autoencoder(object):
    """
    Map an input vector to another space.
    """
    def __init__(self, map_size, activation_function=tanh,
                 dactivation_function=dtanh, denoise=True,
                 normalizer=None):
        """
        arguments:
            layers - list of Layer objects
        """
        self.map_size = map_size
        self.activation_function = activation_function
        self.dactivation_function = dactivation_function
        self.denoise = denoise
        self.normalizer = normalizer
        self.layers = [None] * 3

    def propagate_forward(self, row):
        if self.layers[0].bias:
            self.layers[0].visible[1:] = row
        else:
            self.layers[0].visible = row

        for layer_index in range(1, len(self.layers)):
            layer = self.layers[layer_index]
            layer.propagate_forward(self.layers[layer_index - 1].visible)

    def noise(self, x, p_noise=.5):
        """
        Return a copy of x with values set to 0 for a randomly sized random set
        of features.
        args:
            x - a matrix
            p_noise - the probability of setting any given element
                      to 0

        returns:
            a copy of x with values randomly set to zero.
        """
        data = numpy.array(x)
        for index, row in enumerate(data):
            n_features = len(row)
            while n_features > int(p_noise * len(row)):
                n_features = numpy.random.randint(0, len(row))
            inds = numpy.random.permutation(range(len(row)))[:n_features]
            data[index][inds] = 0.

        return data

    def init_weights(self):
        # the input layer doesn't have weights
        for layer_index in range(1, len(self.layers)):
            layer = self.layers[layer_index]
            layer.init_weights(len(self.layers[layer_index - 1].visible))

    def fit(self, x, iterations=100, noise=.5, shuffle=True):
        """
        args:
            x - a numpy array
            iterations - number of iterations to train
            noise - the proportion of features to randomly set to zero
            shuffle - boolean, shuffle data for each iteration
        """

        self.layers = [nn.layer(x.shape[1], linear, dlinear),
                       nn.layer(self.map_size, self.activation_function,
                                self.dactivation_function,
                                bias=False),
                       nn.layer(x.shape[1], linear, dlinear, bias=False)]

        self.init_weights()

        # sparse filtering
        if self.normalizer:
            self.normalizer = sparse_filtering_normalizer(x)
            data = self.normalizer.normalize(x)
        else:
            data = x

        if self.denoise:
            data = self.noise(data, noise)

        for iteration in range(iterations):
            if shuffle:
                inds = numpy.random.permutation(range(len(data)))
                data = data[inds]

            for row_ind, row in enumerate(data):
                # propagate forward
                self.propagate_forward(row)

                # propagate backward -- the input weights never
                # change
                layer = self.layers[-1]
                target = row

                # do hidden layers
                for layer_index in range(len(self.layers) - 1, 0, -1):
                    layer = self.layers[layer_index]
                    input_layer = self.layers[layer_index - 1]
                    target = layer.propagate_backward(input_layer, target)

    def predict(self, x):
        """
        Use the autoencoder to denoise the input vectors.
        """
        data = as_matrix(x)
        if self.normalizer:
            data = self.normalizer.normalize(data)

        predicted = []
        for row in data:
            self.propagate_forward(row)
            prediction = self.layers[-1].visible
            predicted.append(prediction)

        if self.normalizer:
            return self.normalizer.denormalize(predicted)

        return numpy.array(predicted)

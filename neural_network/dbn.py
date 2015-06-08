from ..base import model
import numpy
from ..util.preprocessing import as_matrix
from autoencoder import autoencoder
import nn


class dbn(model):

    def __init__(self, layers, pretrain=True):
        """
        Initialization of the deep belief network with given layers.

        arguments:
            layers - list of Layer objects
            pretrain - whether or not to pretrain layers with
                       autoencoders
        """
        self.layers = layers
        # initialize hidden layer weights
        self.init_weights()
        self.pretrain = pretrain

    def init_weights(self):
        # the input layer doesn't have weights
        for layer_index in range(1, len(self.layers)):
            layer = self.layers[layer_index]
            layer.init_weights(len(self.layers[layer_index - 1].visible))

    def propagate_forward(self, row):
        if self.layers[0].bias:
            self.layers[0].visible[1:] = row
        else:
            self.layers[0].visible = row

        for layer_index in range(1, len(self.layers)):
            layer = self.layers[layer_index]
            layer.propagate_forward(self.layers[layer_index - 1].visible)

    def fit(self, x, y, iterations=1, shuffle=True):
        data = as_matrix(x)
        response = y

        self.layers[0] = nn.layer(len(data[0]), bias=self.layers[0].bias)

        # initialize the hidden layers with stacked autoencoders
        if self.pretrain:
            out = x
            for index in range(1, len(self.layers) - 1):
                layer = self.layers[index]
                trained = autoencoder(len(layer.visible))
                trained.fit(out)
                out = trained.predict(out)
                layer.weights = trained.layers[1].weights

        errors = []
        for iteration in range(iterations):
            if shuffle:
                inds = numpy.random.permutation(range(len(data)))
                data = data[inds]
                response = response[inds]

            for row_ind, row in enumerate(data):
                # propagate forward
                self.propagate_forward(row)

                # propagate backward -- the input weights never change
                layer = self.layers[-1]
                target = response[row_ind]
                errors.append(sum(target - layer.visible))

                # do hidden layers
                for layer_index in range(len(self.layers) - 1, 0, -1):
                    layer = self.layers[layer_index]
                    input_layer = self.layers[layer_index - 1]
                    target = layer.propagate_backward(input_layer, target)

        return errors

    def predict(self, x):
        data = as_matrix(x)

        predicted = []
        for row in data:
            # propagate forward
            self.propagate_forward(row)
            prediction = self.layers[-1].visible
            predicted.append(prediction)

        return numpy.array(predicted)

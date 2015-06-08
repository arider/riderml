"""
Components for building neural networks.
"""


import numpy
from ..util.linear_functions import linear, dlinear

ETA_PLUS = 1.4
ETA_MINUS = 0.5
MIN_SS = .001
MAX_SS = 10


def propagate_forward(layer, input_values):
    new_values = layer.activation_function(numpy.dot(input_values,
                                                     layer.weights))
    if layer.bias:
        layer.visible[1:] = new_values[1:]
    else:
        layer.visible = new_values


def propagate_backward_irpropm(layer, input_layer, target, batch_size=20):
    """
    Use the iRprop- algorithm for backpropagation with the given batch size.
    args:
        layer - the visible layer
        input_layer - the input layer
        target - the target output vector
        batch_size - how many examples to use to update the gradient before
                     acting on it.
    """
    if input_layer.bias:
        input_layer.visible[0] = 1.

    error = target - layer.visible
    gradient = error * layer.derivative_function(layer.visible)

    change = numpy.outer(input_layer.visible, gradient)

    # adds an attribute to this layer to keep track of how many
    # iterations of backprop have happened since the last update
    # to the weights

    # initialize the layer with backprop count if necessary.
    try:
        layer.backprop_count
    except AttributeError:
        layer.backprop_count = 1
        layer.previous_gradient = layer.gradient
        layer.gradient = numpy.zeros(layer.weights.shape)

    if layer.backprop_count <= batch_size - 1:
        layer.backprop_count += 1
        layer.gradient += change

    if layer.backprop_count >= batch_size - 1:
        signs = numpy.sign(layer.gradient * layer.previous_gradient)
        for ci in range(0, signs.shape[1]):
            for ri in range(0, signs.shape[0]):
                if signs[ri, ci] < 0.:
                    layer.step_size[ri, ci] = max(
                        MIN_SS, ETA_MINUS * layer.step_size[ri, ci])
                    layer.gradient[ri, ci] = 0.
                elif signs[ri, ci] > 0.:
                    layer.step_size[ri, ci] = min(
                        MAX_SS, ETA_PLUS * layer.step_size[ri, ci])

        layer.weights += numpy.sign(layer.gradient) * layer.step_size
        layer.previous_gradient = layer.gradient
        layer.gradient = numpy.zeros(layer.gradient.shape)
        layer.backprop_count = 0

    # hidden deltas
    hidden_change = (numpy.dot(gradient, layer.weights.T)
                     * input_layer.derivative_function(input_layer.visible))
    estimated_hidden = input_layer.visible + hidden_change

    return estimated_hidden


def propagate_backward(layer, input_layer, target):
    """
    Update the weights for this layer.
    """
    if numpy.isscalar(target):
        tmp = target
        target = numpy.zeros([1, 1])
        target[0] = tmp

    error = target - layer.visible
    gradient = error * layer.derivative_function(layer.visible)

    hidden_change = (numpy.dot(gradient, layer.weights.T)
                     * input_layer.derivative_function(input_layer.visible))
    estimated_hidden = input_layer.visible + hidden_change

    change = numpy.outer(input_layer.visible, gradient)

    layer.weights += layer.learning_rate * change
    return estimated_hidden


def fully_connected_weights(in_size, out_size):
    weights = numpy.random.random((in_size, out_size)) - .5
    return weights


class layer():
    """
    Generic neural network layer class.
    """

    def __init__(self, layer_size, activation_function=linear,
                 derivative_function=dlinear,
                 forward_function=propagate_forward,
                 backward_function=propagate_backward_irpropm,
                 init_weights_function=fully_connected_weights, bias=True):
        """
        A neural network layer.

        layer_size - the number of inputs
        activation_function - sigmoid or other
        derivative_function - derivative of activation
        forward_function - the forward propagation function
        backward_funciton - the backpropagation function
        bias - include a bias node
        """
        self.forward_propagation = forward_function
        self.back_propagation = backward_function
        self.activation_function = activation_function
        self.derivative_function = derivative_function
        self.bias = bias

        # the activations of these nodes
        bias_add = 0
        if self.bias:
            bias_add = 1
        self.visible = numpy.ones(layer_size + bias_add)
        self.init_weights_function = init_weights_function

    def init_weights(self, in_size):
        self.weights = self.init_weights_function(in_size,
                                                  len(self.visible))
        if self.bias:
            self.weights[:, 0] = numpy.ones(self.weights.shape[0])

        # for rprop
        self.learning_rate = numpy.ones(self.weights.shape) * .1
        self.previous_gradient = numpy.zeros(self.weights.shape)
        self.gradient = numpy.zeros(self.weights.shape)
        self.step_size = numpy.ones(self.weights.shape) * .1

    def propagate_forward(self, values):
        return self.forward_propagation(self, values)

    def propagate_backward(self, input_layer, target):
        return self.back_propagation(self, input_layer, target)

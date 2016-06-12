import numpy

ETA_PLUS = 1.2
ETA_MINUS = 0.5


def stochastic_gradient_descent(function,
                                derivative,
                                x, y,
                                theta=None,
                                iterations=100,
                                learning_rate=0.000001,
                                shuffle=True,
                                batch_size=.2):
    """
    Gradient descent with mini batches.  Batch_size is a float for proportion
    of the data or an int.  1 means standard stochastic gradient descent.

    args:
        function - a function taking two arrays and returning one
        derivative - derivative of the function
        x - a numpy array; instances in rows
        y - a numpy array
        theta - initial coefficients.
        iterations - number of iterations to do
        learning_rate - initial learning rate
        shuffle - whether or not to shuffle the data before each iteration
        batch_size - proportion or integer size of batches.
    """

    if theta is None:
        theta = numpy.random.rand(x.shape[1], y.shape[1])

    assert x.shape[1] == theta.shape[0]

    # translate float into batch size int
    batch_number = float(batch_size)
    if batch_size < 1 and batch_size > 0:
        batch_number = int(batch_size * x.shape[0])
    if batch_number < 1:
        batch_number = 1

    # initialize feature specific learning rates
    delta = numpy.zeros(theta.shape)
    delta += learning_rate
    previous_gradient = numpy.zeros([x.shape[1], theta.shape[1]])

    current_theta = numpy.array(theta)
    for iteration in range(iterations):

        # shuffle data
        if shuffle:
            inds = numpy.random.permutation(range(x.shape[0]))
            x = x[inds]
            y = y[inds]

        # process batches
        batch_index = 0
        for i in range(int(x.shape[0] / batch_number)):
            if i == int(x.shape[0] / batch_number) - 1:
                batch_inds = range(int(batch_index * batch_number), x.shape[0])
            else:
                batch_inds = range(int(batch_index * batch_number),
                                   int((batch_index + 1) * batch_number))

            batch_x = x[batch_inds]
            batch_y = y[batch_inds]

            loss = function(batch_x, current_theta) - batch_y

            # avg gradient per example
            gradient = (derivative(batch_x, theta).T.dot(loss) /
                        batch_x.shape[0])

            # update the learning rate
            sign = numpy.sign(gradient * previous_gradient)
            for ci in range(sign.shape[1]):
                for f in range(sign.shape[0]):
                    if sign[f, ci] < 0.:
                        delta[f, ci] = ETA_MINUS * delta[f, ci]
                        gradient[f, ci] = 0.
                    elif sign[f, ci] > 0.:
                        delta[f, ci] = ETA_PLUS * delta[f, ci]

            current_theta -= numpy.sign(gradient) * delta
            previous_gradient = gradient

            batch_index += 1
    return current_theta


def gradient_descent(function,
                     derivative,
                     x, y,
                     theta=None,
                     iterations=100,
                     learning_rate=0.000001,
                     shuffle=True):
    """
    Gradient descent -- use irprop- algorithm to adjust learning rate on a
    per-feature basis

    arguments:
        function - the function to learn parameters of (takes (x, theta))
                   ex: logistic, linear, etc....
        derivative - the derivative of the function
        x - the input data in a matrix at least (1, 1)
        y - the response variable(s)
        theta - coefficients array
        iterations - number of iterations
        learning_rate - the learning rate, float
        shuffle - permute the data at each iteration
    """
    if theta is None:
        theta = numpy.random.rand(x.shape[1], y.shape[1])

    # parameters for rprop
    previous_gradient = numpy.zeros([x.shape[1], theta.shape[1]])
    delta = numpy.zeros(theta.shape)
    delta += learning_rate

    for i in range(0, int(iterations)):
        if shuffle:
            inds = numpy.random.permutation(range(x.shape[0]))
            x = x[inds]
            y = y[inds]

        # avg gradient per example
        loss = function(x, theta) - y
        gradient = derivative(x, theta).T.dot(loss) / x.shape[0]

        # update the learning rate
        sign = gradient * previous_gradient
        for ci in range(sign.shape[1]):
            for f in range(sign.shape[0]):
                if sign[f, ci] < 0.:
                    delta[f, ci] = ETA_MINUS * delta[f, ci]
                    gradient[f, ci] = 0.
                elif sign[f, ci] > 0.:
                    delta[f, ci] = ETA_PLUS * delta[f, ci]

        theta -= numpy.sign(gradient) * delta
        previous_gradient = gradient

    return theta


def adagrad(function, d_function, x, y, theta, iterations,
            learning_rate=0.01, shuffle=True, smoothing=.5):
    """
    Gradient descent -- use rprop algorithm to adjust learning rate on a
    per-feature basis

    arguments:
        function - the function to learn parameters of (takes x, theta)
        derivative - the derivative of the function
            ex: logistic, linear, etc....
        x - the input data in a matrix at least (1, 1)
        y - the response variable(s)
        theta - coefficients array
        iterations - number of iterations
        learning_rate - the learning rate, float
        shuffle - permute the data at each iteration
        smoothing - exponential smoothing in case adagrad is too
                    aggressive in step size
    """
    running_gradient = numpy.zeros(theta.shape)
    for iteration in range(iterations):
        loss = function(x, theta) - y
        gradient = loss.T.dot(d_function(x)) / x.shape[0]
        # the step size is too aggressive with 'canonical' adagrad on
        # non-sparse problems, so we use exponential smoothing instead of
        # running_gradient += gradient ** 2
        if smoothing:
            running_gradient = (smoothing * running_gradient +
                                (1 - smoothing) * (gradient ** 2).T)
        else:
            running_gradient += gradient ** 2

        lr = numpy.multiply(1. / (numpy.sqrt(running_gradient)), gradient.T)
        theta -= learning_rate * lr
    return theta

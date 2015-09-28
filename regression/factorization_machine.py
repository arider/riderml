from ..base import model
import numpy
from scipy.sparse import csr_matrix


# irprop- jump size
ETA_MINUS = .5
ETA_PLUS = 1.2


class factorization_machine(model):

    def __init__(self, n_factors=10, model_type='regression', debug=False):

        self.n_factors = n_factors
        # list of factor matrices: one per matrix in x
        self.feature_weights = None
        # list of weight matrices: one per matrix in x
        self.factor_weights = None
        self.learning_rate = .01
        self.debug = debug

    def update_learning_rates(self, gradient, previous_gradient, delta,
                              weights):
        """
        irprop- for the given part of the model

        args:
            gradient - the gradient
            previous_gradient - ...
            delta - the current step sizes
            weights - the weights

        returns:
            updated weights, updated step sizes
        """
        sign = numpy.multiply(gradient, previous_gradient)
        for ri in xrange(sign.shape[0]):
            for ci in xrange(sign.shape[1]):
                if sign[ri, ci] < 0.:
                    delta[ri, ci] = (
                        ETA_MINUS * delta[ri, ci])
                    gradient[ri, ci] = 0.
                elif sign[ri, ci] > 0.:
                    delta[ri, ci] = (
                        ETA_PLUS * delta[ri, ci])

        sign = numpy.sign(gradient.copy())

        weights = (weights - numpy.multiply(sign, delta))

        return weights, delta

    def fit(self, data, y, iterations=1, regularization_prop=0.,
            batch_prop=0.):
        """
        args:
            data - a sparse design matrix
            y - the target variable
            iterations - number of iterations of SGD to run
            regularization_size - the proportion of the data to use for
                                  adaptive regularization
            batch_prop - proportion of data size to use in batches --
                         calculated after the validation set for regularization
                         is removed.
        """

        # add the bias
        x = csr_matrix((data.shape[0], data.shape[1] + 1))
        x[:, 0] = 1
        x[:, 1:] = data

        self.feature_weights = numpy.random.rand(1, x.shape[1])
        self.factor_weights = numpy.random.rand(self.n_factors, x.shape[1])

        if regularization_prop >= 1. or regularization_prop < 0.:
            raise ValueError('Bad value for regularization size')

        # get the test set for adaptive regularization
        test_size = int(regularization_prop * x.shape[0])

        # get the size of the batches for batch SGD
        if batch_prop < 1 and batch_prop > 0.:
            batch_size = int(batch_prop * (x.shape[0] - test_size))
        elif batch_prop == 0:
            batch_size = (x.shape[0] - test_size)
        else:
            batch_size = int(batch_size)

        # feature step sizes
        previous_feature_gradient = numpy.zeros(self.feature_weights.shape)
        feature_delta = (numpy.ones(self.feature_weights.shape) *
                         self.learning_rate)

        # factor step sizes
        previous_factor_gradient = numpy.zeros(self.factor_weights.shape)
        factor_delta = (numpy.ones(self.factor_weights.shape) *
                        self.learning_rate)

        for iteration in range(iterations):
            if self.debug:
                print "Iteration", iteration

            # shuffle and choose validation set if we're doing adaptive
            # regularization
            inds = numpy.random.permutation(x.shape[0])
            if test_size > 0:
                test_inds = inds[:test_size]
                inds = inds[test_size:]
                train_x = x[inds]
                train_y = y[inds]
                test_x = x[test_inds]
                test_y = x[test_inds]
            else:
                train_x = x[inds]
                train_y = y[inds]
                test_x = []
                test_y = []

            y_hat = self.predict(x[:, 1:])
            loss = y_hat - y

            # Do the feature weight update
            feature_gradient = self.update_features_step(x,
                                                         y,
                                                         loss)

            weights, delta = self.update_learning_rates(
                feature_gradient, previous_feature_gradient, feature_delta,
                self.feature_weights)
            self.feature_weights = weights
            feature_delta = delta

            previous_feature_gradient = feature_gradient

            # Do the factor weight updates
            factor_gradient = self.update_factors_step(x,
                                                       y,
                                                       loss)

            weights, delta = self.update_learning_rates(
                factor_gradient, previous_factor_gradient, factor_delta,
                self.factor_weights)

            self.factor_weights = weights
            factor_delta = delta

            previous_factor_gradient = factor_gradient

#            regularization_step(self, x, y):

    def update_features_step(self, instances, targets, loss):
        """
        Update the linear part of the model.

        args:
            instances - a sparse matrix
            targets - a sparse matrix of target values
            loss - the loss

        returns:
            the gradient of the linear part of the model
        """
        # Do the feature weight update
        # average gradient
        feature_gradient = (csr_matrix(loss).dot(instances) /
                            instances.shape[0]).todense()[0]

        return feature_gradient

    def update_factors_step(self, instances, targets, loss):
        """
        Update the factor part of the model.

        args:
            instances - a sparse matrix
            targets - a sparse matrix of target values
            loss - the loss

        returns:
            the gradient of the factor part of the model
        """
        # Do the factor weight updates
        factor_gradient = numpy.zeros(self.factor_weights.shape)
        for instance_index, instance in enumerate(instances):
            for feature_count_index in range(len(instance.nonzero()[1])):
                for factor_index in range(self.factor_weights.shape[0]):
                    first_part = (
                        instance.data[feature_count_index] *
                        self.factors_over_features[instance_index,
                                                   factor_index])
                    second_part = (self.factor_weights[factor_index,
                                                       feature_count_index] *
                                   instance.data[feature_count_index] ** 2)
                    factor_gradient[factor_index, feature_count_index] += (
                        loss[0, instance_index] *
                        (first_part - second_part))

        factor_gradient /= instances.shape[0]

        return factor_gradient

    def regularization_step(self, x, y):
        """
        Use the test set to determine good regularization parameters.
        """
        raise NotImplementedError("Not yet implemented!")

    def predict(self, data):
        # add the bias
        x = csr_matrix((data.shape[0], data.shape[1] + 1))
        x[:, 0] = 1
        x[:, 1:] = data

        # the linear part
        linear_part = csr_matrix(self.feature_weights).dot(x.T).todense()[0]

        # V_jfX_j
        self.factors_over_features = x.dot(self.factor_weights.T)

        # the factor part
        # square x and factor_weights
        sq_x = x.copy()
        sq_x.data **= 2
        sq_factor_weights = self.factor_weights.copy() ** 2

        # X^2V^2t
        factors_over_features_sq = sq_x.dot(sq_factor_weights.T)
        first_part = self.factors_over_features.copy() ** 2

        # sum over k
        factor_part = .5 * (first_part
                            - factors_over_features_sq).sum(axis=1)

        # the whole thing
        predicted = numpy.array(linear_part + factor_part.T)

        return predicted

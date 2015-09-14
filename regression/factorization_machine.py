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

    def fit(self, data, y, iterations=1):
        # add the bias
        x = csr_matrix((data.shape[0], data.shape[1] + 1))
        x[:, 0] = 1
        x[:, 1:] = data

        self.feature_weights = numpy.random.rand(1, x.shape[1])
        self.factor_weights = numpy.random.rand(self.n_factors, x.shape[1])

        # feature step sizes
        previous_feature_gradient = numpy.zeros(self.feature_weights.shape)
        feature_delta = numpy.ones(self.feature_weights.shape) * self.learning_rate

        # factor step sizes
        previous_factor_gradient = numpy.zeros(self.factor_weights.shape)
        factor_delta = numpy.ones(self.factor_weights.shape) * self.learning_rate

        for iteration in range(iterations):
            if self.debug:
                print "Iteration", iteration

            y_hat = self.predict(x[:, 1:])
            loss = y_hat - y

            # Do the feature weight update
            # average gradient
            feature_gradient = (csr_matrix(loss).dot(x) / x.shape[0]).todense()[0]

            sign = numpy.multiply(feature_gradient, previous_feature_gradient)
            for ri in xrange(sign.shape[0]):
                for ci in xrange(sign.shape[1]):
                    if sign[ri, ci] < 0.:
                        feature_delta[ri, ci] = ETA_MINUS * feature_delta[ri, ci]
                        feature_gradient[ri, ci] = 0.
                    elif sign[ri, ci] > 0.:
                        feature_delta[ri, ci] = ETA_PLUS * feature_delta[ri, ci]

            sign = numpy.sign(feature_gradient.copy())
            
            self.feature_weights = (self.feature_weights
                                    - numpy.multiply(sign, feature_delta))
            previous_feature_gradient = feature_gradient.copy()
                
            # Do the factor weight updates
            factor_gradient = numpy.zeros(self.factor_weights.shape)
#            rind, cind = x.nonzero()
            for instance_index, instance in enumerate(x):
                for feature_index in instance.nonzero()[1]:
                    for factor_index in range(self.factor_weights.shape[0]):
                        first_part = instance.data[feature_index] * self.factors_over_features[instance_index, factor_index]
                        second_part = self.factor_weights[factor_index, feature_index] * instance.data[feature_index] ** 2
                        factor_gradient[factor_index, feature_index] += (loss[0, instance_index] * (first_part - second_part))

            factor_gradient /= x.shape[0]

            sign = numpy.multiply(factor_gradient, previous_factor_gradient)
            for ri in xrange(sign.shape[0]):
                for ci in xrange(sign.shape[1]):
                    if sign[ri, ci] < 0.:
                        factor_delta[ri, ci] = ETA_MINUS * factor_delta[ri, ci]
                        factor_gradient[ri, ci] = 0.
                    elif sign[ri, ci] > 0.:
                        factor_delta[ri, ci] = ETA_PLUS * factor_delta[ri, ci]

            sign = numpy.sign(factor_gradient)

            self.factor_weights = (self.factor_weights
                                   - numpy.multiply(sign, factor_delta))
            previous_factor_gradient = factor_gradient.copy()


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

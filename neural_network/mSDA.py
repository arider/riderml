import numpy


class mSDA():
    def __init__(self, squashing_function=numpy.tanh,
                 reverse_squashing_fuction=numpy.arctanh,
                 debug=False):
        # the mDAs that this mSDA is made of
        self.layers = []
        # the current values of the hidden layers
        self.values = []
        self.squashing_function = squashing_function
        self.reverse_squashing_function = reverse_squashing_fuction
        self.debug = debug

    def fit(self, x, noise, n_layers):
        """
        Stack mDAs to learn a nonlinear denoising autoencoder.
        args:
            x - numpy array
            noise - either a number in [0, 1] or a list or numpy array of them
            n_layers - number of layers (mDAs) to use
        """
        previous_input = x

        for layer_index in range(n_layers):
            if self.debug:
                print 'layer', layer_index
            layer = mDA()
            try:
                iter(noise)
                layer.fit(previous_input, noise[layer_index])
            except TypeError:
                layer.fit(previous_input, noise)

            out = layer.encode(previous_input)
            self.values.append(out)
            self.layers.append(layer)
            previous_input = out

    def encode(self, x):
        """
        Run x through the stack of mDAs.

        args:
            x - ndarray
        """
        i = 0
        previous_output = x
        for layer in self.layers:
            if self.debug:
                print "encoding layer {}: {}".format(i, previous_output)
            i += 1
            previous_output = layer.encode(previous_output)
        if self.debug:
            print "encoding layer {}: {}".format(i, previous_output)

        return previous_output

    def decode(self, x):
        """
        Take a transformed input (so, the output) and reverse the post-training
        transformation the reverse the normalization.

        args:
            x - ndarray
        """
        previous_output = x
        i = 0
        for layer in self.layers[::-1]:
            if self.debug:
                print "decoding layer {}: {}".format(i, previous_output)
            i += 1
            previous_output = layer.decode(previous_output)
        if self.debug:
            print "decoding layer {}: {}".format(i, previous_output)

        return previous_output


class mDA():
    def __init__(self, squashing_function=numpy.tanh,
                 reverse_squashing_fuction=numpy.arctanh, debug=False):
        self.weights = None
        self.squashing_function = squashing_function
        self.reverse_squashing_function = reverse_squashing_fuction
        self.debug = debug

    def fit(self, x, noise, smooth=0.):
        """
        Fit the denoised autoencoder with OLS.

        OLS:

        W = PQ-1, Q = X_tild X_tild^T P = X_bar X_tild^T

        where X_tild is the noised version of X and X_bar is the m-times
        repeated X

        mDA uses the expected value E[P] and E[Q] calculated by using the
        probability of the features alpha and beta both not being noised as (1
        - p)^2 nd taking the product with the scatter matrix XX^T

        args:
            x - numpy array
            p - a float in [0, 1]
        """
        # add the bias
        X = numpy.ones((x.shape[0], x.shape[1] + 1))
        X[:, :-1] = x

        # E[Q]
        q = numpy.outer(numpy.ones(X.shape[1]), (1 - noise))
        q[-1] = 1

        # scatter matrix
        S = numpy.dot(X.T, X)

        # E[Q]
        eq = S * q * q.T
        # correct for diagonal
        for i in range(len(eq)):
            eq[i, i] = q[i] * S[i, i]

        # E[P]
        ep = S[:, :-1] * q

        # calculate weights
        reg = smooth * numpy.identity(S.shape[0])
        reg[-1, -1] = 0.

        W, _, _, _ = numpy.linalg.lstsq((eq + reg), ep)
        W = W.T
        self.weights = W

    def encode(self, x):
        """
        Get the output.

        args:
            x - a matrix without the bias.
            squash_function - function to apply to the predictions
        """
        if len(x.shape) == 1:
            X = numpy.ones(len(x) + 1)
            X[:-1] = x
        else:
            X = numpy.ones((x.shape[0], x.shape[1] + 1))
            X[:, :-1] = x

        out = self.squashing_function(numpy.dot(self.weights, X.T)).T
        if self.debug:
            print "SQUASHED", out
        for ri in range(len(out)):
            for ci in range(len(out[ri])):
                if out[ri, ci] >= 1.:
                    out[ri, ci] = .99999
                if out[ri, ci] <= -1.:
                    out[ri, ci] = -.99999

        return out

    def decode(self, x):
        """
        take the output and reverse the squashing function

        returns:
            a numpy array
        """
        decoded = self.reverse_squashing_function(x)
        if self.debug:
            print "DECODED", decoded

        decoded = numpy.nan_to_num(decoded)
        return decoded

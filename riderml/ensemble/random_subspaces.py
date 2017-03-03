from ..base import ensemble
import numpy
from ..util.preprocessing import as_matrix
from sklearn.metrics import accuracy_score


class random_subspaces(ensemble):
    """
    A random subspace ensemble of models.
    """

    def __init__(self, model_class, init_args, fit_args, n_models,
                 subspace_size, response_matching=False):
        """
        model_class - the base learner model class
        init_args - arguments to initialize the component models
        fit_args - arguments to fit the component models
        n_models - the number of models to train
        subspace_size - if < 1 and > 0 then a proportion of the feature space,
                        if == 0, then a randomproporiton of the feature space
                                 for each component model,
                        if >= 1, an integer number of features.
        response_matching - indicates whether the response features are the
                            same as the input for timeseries.  If True then
                            each base learner only makes predictions on
                            features that is has as input. If False then all
                            base learners are predicting the same set of
                            features.
        """
        # list of models
        self.models = [model_class(**init_args) for i in range(n_models)]
        # feature set for each model
        self.n_models = n_models
        self.model_features = [set()] * n_models
        # subspaces size for whole algorithm
        self.subspace_size = subspace_size
        self.response_matching = response_matching
        self.init_args = init_args
        self.fit_args = fit_args
        self.model_class = model_class

    def select_features(self, x):
        # select features
        # select number of features
        n_features = 0
        if self.subspace_size == 0:
            if numpy.atleast_2d(x).shape[1] == 1:
                return [0]
            n_features = numpy.random.randint(1, numpy.atleast_2d(x).shape[1])
        if self.subspace_size < 1 and self.subspace_size > 0:
            n_features = int(x.shape[1] * self.subspace_size)
        if self.subspace_size >= 1:
            n_features = self.subspace_size

        return numpy.random.permutation(range(x.shape[1]))[:n_features]

    def fit(self, x, y, model_inds=None, async=False):
        """
        Fit the base learners.  This assumes that if you have a
        multidimentional response that every learner is predicting every
        response feature.
        x - matrix
        y - response variables
        model_inds - a list of indices -- if given then train only the models
                     with given index.  This is for retraining
        async - boolean whether or not to parallelize component model training
        """
        self.output_shape = 1
        if len(y.shape) > 1:
            self.output_shape = y.shape[1]

        if not model_inds:
            model_inds = range(len(self.models))

        # initialize all models
        for model_index in model_inds:
            self.models[model_index] = self.model_class(**self.init_args)

        for model_index in model_inds:
            features = self.select_features(x)
            self.model_features[model_index] = set(features)

            if self.response_matching:
                self.models[model_index].fit(x[:, features],
                                             y[:, features],
                                             **self.fit_args)
            else:
                self.models[model_index].fit(x[:, features], y,
                                             **self.fit_args)

    def predict_components(self, x):
        """
        Get predictions for each model and return the results not
        aggregated.  This is for use in HUWRS where we want to
        measure whether the model error has increased or the
        dependent concept has changed.

        args:
            x - matrix
        returns:
            results - matrix
            count - numpy array of counts over the output
                    variables (number of predictors each
                    feature is aggregated over)
        """
        count = numpy.zeros(self.output_shape)
        results = []
        for model_index, model in enumerate(self.models):
            predictions = model.predict(
                x[:, list(self.model_features[model_index])])
            if self.response_matching:
                indices = list(self.model_features[model_index])
                results.append(predictions)
                count[indices] += 1
            else:
                results.append(predictions)
                count += 1

        # Get mean predictions
        return results, count

    def predict(self, x):
        """
        calculate aggregate predictions.

        returns:
            aggregate - the mean prediction
            count - the number of models contributing to each predicted feature
        """
        count = numpy.zeros(self.output_shape)
        aggregate = numpy.zeros([x.shape[0], self.output_shape])
        for model_index, model in enumerate(self.models):
            predictions = model.predict(
                x[:, list(self.model_features[model_index])])
            if self.response_matching:
                indices = list(self.model_features[model_index])
                aggregate[:, indices] += as_matrix(predictions)
                count[indices] += 1
            else:
                aggregate += as_matrix(predictions)
                count += 1

        # Get mean predictions
        aggregate /= count
        return aggregate, count

    def score(self, x, y):
        predicted = self.predict(x)
        actual = y

        return accuracy_score(actual, predicted)

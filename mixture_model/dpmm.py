import numpy
import random
import time
from scipy.special import gammaln

# TODO: remove everything about multinomials in here. Pass in a distribution
# object with functions for sampling etc...


class multinomial():
    """
    Class for keeping track of the parameters of a single multinomial.
    """
    def __init__(self, number_features, smooth=.0000001):
        # the probabilities
        self.phi = numpy.zeros([number_features])
        # counts
        self.n = 0
        self.smooth = smooth

    def posterior(self, instance):
        """
        args:
            instance - an observation of the PMF
        """
        ps = (self.phi + self.smooth) / numpy.sum(self.phi + self.smooth)
        p = gammaln(numpy.sum(instance) + 1)
        p -= numpy.sum(gammaln(instance + 1))
        p += numpy.sum(instance * numpy.log(ps))

        return p

    def rem_i(self, instance):
        self.n -= 1
        self.phi -= instance

    def add_i(self, instance):
        self.n += 1
        self.phi += instance


class gaussian():
    def __init__(self, number_features, smooth=.0000001):
        """
        args:
            number_features - number of features to expect for initialization
        """
        self.data = []
        self.sigma = numpy.zeros([number_features, number_features])
        self.mu = numpy.zeros(number_features)
        # the number of instances in this component
        self.n = 0
        self.smooth = smooth

    def update_params(self):
        if len(self.data) == 1:
            self.sigma = numpy.array([self.n ** (-1. / (len(self.mu) + 4))])
        elif len(self.data) > 1:
            self.sigma = numpy.cov(numpy.array(self.data).T)
            self.mu = numpy.array(self.data).sum(axis=0) / float(len(self.data))
        else:
            self.mu = [0] * len(self.mu)


    def posterior(self, instance):
        """
        args:
            instance - an observation of the PMF
        returns:
            the posterior probability
        """
        if len(self.sigma) == 1:
            ll = - .5 * numpy.log(self.sigma[0] ** len(self.mu))
            dotpart = .5 * (instance - self.mu) * (1 / self.sigma[0])
            ll -= numpy.dot(dotpart, (instance - self.mu))
            ll -= self.n / 2 * numpy.log(numpy.pi)

            return ll

        determinant = numpy.linalg.det(self.sigma)
        if determinant <= 0:
            # use kde
            total_ll = 0.
            bandwidth = self.n ** (-1. / (len(self.mu) + 4))
            for kernel in self.data:
                ll = - .5 * numpy.log(bandwidth ** len(self.mu))
                dotpart = .5 * (instance - self.mu) * (1 / bandwidth)
                ll -= numpy.dot(dotpart, (instance - self.mu))
                ll -= self.n / 2 * numpy.log(numpy.pi)
                total_ll += ll

            return total_ll / self.n

        # determinant is positive and non-negative; do normal pdf
        norm_coeff = (len(self.sigma)
                      * numpy.log(2 * numpy.pi)
                      + numpy.log(determinant))
        error = instance - self.mu
        numerator = numpy.linalg.solve(self.sigma, error).T.dot(error)

        return - 0.5 * (norm_coeff + numerator)

    def rem_i(self, instance):
        self.n -= 1

        # remove the instance from the data, recalculate sigma
        if tuple(instance) in self.data:
            self.data.pop(self.data.index(tuple(instance)))
        self.update_params()

    def add_i(self, instance):
        self.n += 1

        self.data.append(tuple(instance))
        self.update_params()


class dpmm:
    """
    Dirichlet process mixture model with gibbs sampling approach to
    sampling for non-conjugate distributions. See alg. 8 in Neal 2000.
    """
    def __init__(self, data, alpha, smooth, m,
                 base_distribution=multinomial, debug=False):
        """
        args:
            data - a list of arrays or a 2d array
            alpha - parameter to the dirichlet
            smooth - float for laplace smoothing for LL calculation
            m - number of auxiliary parameters
        """
        self.debug = debug
        self.base_distribution = base_distribution
        self.btime = 0
        # the number of features
        self.nf = len(data[0])
        self.alpha = float(alpha)
        self.N = len(data)
        self.smooth = float(smooth)
        self.data = data

        self.c = numpy.array([None] * self.N)
        # component membership for each instance
        self.components = []
#        for i in range(self.N):
#            self.add_i(numpy.random.randint(0, n_init_components), i)

        self.new_components = []

        # initialize
        # the number of samples averaged to approximate the integral over F *
        # dG(phi)
        self.m = m
        self.add_i(0, 0)
        self.add_i(0, 1)

    def fit(self, iterations):
        """
        The gibbs sampling loop.
        """
        output = []
        if self.debug:
            print "starting"
        for n in xrange(iterations):
            # sample new once per iteration
            self.new_components = []
            for i in xrange(self.m):
                self.add_rand_phi()

            for i in xrange(self.N):
                self.sample_c(i)

            ll = self.loglikelihood()
            output.append(
                [str(n), str([c.n for c in self.components]), str(ll)])
            if self.debug:
                print "iteration ", n, [c.n for c in self.components], ll
        return output

    def sample_c(self, i):
        """
        Sample component for instance i.
        """
        self.rem_i(i)

        p = self.pc_x(i)
        new_c = numpy.random.multinomial(1, p).argmax()

        self.add_i(new_c, i)

    def loglikelihood(self):
        """
        Calculate the LL of the current state
        """
        # the coefficients
        co = []
        ll = 0
        for ci in range(len(self.components)):
            co.append(self.components[ci].n / (self.N - 1 + self.alpha))
#            co.append(1.)

        # model fit
        for i in range(len(self.data)):
            c = self.c[i]
            p = self.components[c].posterior(self.data[i])
            if self.debug:
                if numpy.isnan(p):
                    print "NAN", self.components[c].__dict__, ll, p
                    import sys
                    sys.exit()

            ll += p
        ll += numpy.sum(co)

        return ll

    def pc_x(self, ind):
        """
        Calculate the probability of each component given the instance at ind

        args:
            ind - index in self.data
        """
        elapsed = time.time()
        p = []
        new_co = (numpy.log(self.alpha) - numpy.log(self.m)
                  - numpy.log((self.N - 1 + self.alpha)))

        for c in range(len(self.components)):
            co = (numpy.log(self.components[c].n)
                  - numpy.log((self.N - 1 + self.alpha)))
            lp = self.components[c].posterior(self.data[ind])
            p.append(lp + co)

        for c in range(self.m):
            lp = self.new_components[c].posterior(self.data[ind])
            p.append(lp + new_co)

        # normalize p
        p -= numpy.max(p)
        p = numpy.exp(p)
        p /= numpy.sum(p)

        if self.btime:
            print "pc_x", time.time() - elapsed
        return p

    def add_rand_phi(self, size=10, random_size=True):
        """
        Take a random sample of instances and create a component from them.

        args:
            size - the number of instances to use to create the component. If
                   random then the maximum size.
            random_size - select a random number of instances in [1, size] to
                          create a component.
        """

        elapsed = time.time()

        if random_size:
            size = random.randint(1, size)

        inds = numpy.random.permutation(range(self.N))[:size]
        self.new_components.append(self.base_distribution(self.nf))

        for instance in self.data[inds]:
            self.new_components[-1].add_i(instance)

        if self.btime:
            print "add_rand_phi", time.time() - elapsed

    def rem_i(self, i):
        elapsed = time.time()
        c = self.c[i]

        if c is None:
            return None

        if self.components[c].n == 1:
            # if this was not the last listed component, change indices in
            if c < len(self.components) - 1:
                ind = numpy.where(self.c > c)
                self.c[ind] = self.c[ind] - 1
            self.components.pop(c)
        else:
            self.components[c].rem_i(self.data[i])

        if self.btime:
            print "rem_i", time.time() - elapsed

    def add_i(self, c, i):
        elapsed = time.time()
        # new component
        if c >= len(self.components):
            self.c[i] = len(self.components)
            self.components.append(self.base_distribution(self.nf))
            self.components[self.c[i]].add_i(self.data[i])
        else:
            self.c[i] = c
            self.components[c].add_i(self.data[i])

        if self.btime:
            print "add_i", time.time() - elapsed

    def get_labels(self, data, random_assign=False):
        """
        Get the component/cluster labels of the given data.

        The case where you don't randomly assign labels based on true model
        probablilities (including dirichlet prior) is akin to a step of EM.

        args:
            data - a numpy array of int
            random_assign - whether to take the most likely component or sample

        returns:
            list of ints
        """
        labels = []
        for i, instance in enumerate(data):
            ps = numpy.zeros(len(self.components))
            for c, component in enumerate(self.components):
                ps[c] = component.posterior(instance)
            ps /= ps.sum()

            new_c = ps.argmax()
            if random_assign:
                new_c = numpy.random.multinomial(1, ps).argmax()

            labels.append(new_c)
        return labels

    def update_data(self, sample, random_assign=True):
        """
        Update the data and assign new data to existing components. For use in
        bagging.

        args:
            sample - a data set numpy array of int
            random_assign - assign points to a randomly selected component
                            based on their likelihood
        """
        self.data = sample
        self.N = len(self.data)
        for i, c in enumerate(self.get_labels(sample, random_assign)):
            self.add_i(c, i)

        # remove components that have no members
        c = 0
        while c < len(self.components):
            if len(numpy.where(self.c == c)[0]) == 0:
                ind = numpy.where(self.c > c)[0]
                self.c[ind] = self.c[ind] - 1
                self.components.pop(c)
            else:
                c += 1

        # recalculate all components
        for c in range(len(self.components)):
            component_instances = sample[numpy.where(self.c == c)]
            self.components[c].phi = numpy.zeros([self.nf])
            self.components[c].n = 0

            for ci in component_instances:
                self.components[c].add_i(ci)
        if self.debug:
            print "INITIALIZED", [c.n for c in self.components]

    def fit_bagging(self, all_data, iterations, bag_size=.1, n_bags=10,
                    random_assign=True):
        """
        Execute gibbs sampling on bagged samples of the data. Instances are
        assigned to the most likely existing (or sampled) component between
        bags, thus ensuring a continuation of the same model.

        args:
            iterations - number of iterations
            bag_size - fraction of data set size if in (0, 1), else just the
                       number of instances
            n_bags - how many iterations of bagging
            random_assign - whether to use the most likely or a sampled
                            component for assigning the new samples to the
                            existing model
        """
        b_size = bag_size
        if 0 < bag_size < 1:
            b_size = int(len(all_data) * bag_size)

        for s in range(n_bags):
            sample = all_data[numpy.random.permutation(
                range(len(all_data)))[:b_size]].astype(int)
            self.update_data(sample, random_assign)
            self.fit(iterations)


if __name__ == "__main__":
    print "TEST RANDOM SIMPLE MULTINOMIALS"
    data = []
    n_draws = 5
    for i in range(50):
        data.append(numpy.random.multinomial(
            n_draws, [.5, .2, .1, .05, .03, .02], size=1)[0])
    for i in range(50):
        data.append(numpy.random.multinomial(
            n_draws, [.02, .03, .05, .1, .2, .5], size=1)[0])
#    for i in range(50):
#        data.append(numpy.random.multinomial(
#            n_draws, [.1, .1, .3, .3, .1, .1], size=1)[0])
    data = numpy.array(data)

    model = dpmm(data, .1, .000000001, 1, 5)
    pre_fit = model.get_labels(data)

    import time
    start = time.time()
    model.fit(50)
#    model.fit_bagging(data, 50, .5, 10)
    print "elapsed", time.time() - start

    print "pre", pre_fit
    print "post", model.get_labels(data)

#    print "TEST BAGGING"
#    model.fit_bagging(data, 50, .5, 10)
#    print "elapsed", time.time() - start
#
#    print "pre", pre_fit
#    print "post", model.get_labels(data)

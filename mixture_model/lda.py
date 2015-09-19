import numpy
from collections import Counter


class lda():
    """
    In LDA each 'document' is composed of words that each belong to some topic.
    Each topic is composed of words. Therefore, we keep track of:
    dt - the count of words in each document that belong to each topic
    tw - the count of words across documents that belong to each topic
    Since we are doing gibbs sampling, we will keep track of the current word
    topic assignments for each document.
    dwt - the topic assignment of each word in every document
    """

    def word_counts_to_indices(self, data):
        """
        Translate the input to list of lists of word counts to lists of word
        index.  Initialize self.dwt
        """
        out = []
        for row in data:
            row_values = []
            for word_index, count in enumerate(row):
                for i in range(count):
                    row_values.append(word_index)
            out.append(row_values)
        return out

    def __init__(self, data, n_topics, n_words, alpha=1., beta=1.):
        """
        args:
            data - a matrix containing multinomial rows (counts of words)
            n_topics - number of components to fit
            n_words - the number of words total
            alpha - dirichlet parameter over documents
            beta - dirichlet parameter over topics
        """
#        # the current topic of each word in each document
        self.dwt = []

        # the data - numpy array
        self.data = self.word_counts_to_indices(data)
        # dirichlet parameter over the documents
        self.alpha = alpha
        # dirichlet parameter over the topics
        self.beta = beta
        # the count of words in each document that belong to each topic
        self.dt = numpy.zeros((len(data), n_topics)) + alpha
        # the count of words across documents that belong to each topic
        self.tw = numpy.zeros((n_topics, data.shape[1])) + beta
        # the number of documents in each topic
        self.td_count = numpy.zeros(n_topics)
        # keep the count of words in each topic for convenience
        self.n_t = numpy.zeros((n_topics,)) + beta

        # initialize
        self.dwt = []
        for d, doc in enumerate(self.data):
            self.dwt.append([0] * len(doc))
            for w, word_index in enumerate(doc):
                p = self.tw[:, word_index] * self.dt[d, :] / self.n_t
                t = numpy.random.multinomial(1, p / p.sum()).argmax()
                self.add_dwt(d, word_index, w, t)

    
    def sample_topics(self, d, word_index):
        """
        Calculate the probability of drawing this word/document from
        each topic.
        """
        probs = ((self.alpha * self.dt[d, :]) *
                 (self.beta + self.tw[:, word_index]) /
                 (len(self.data) * self.beta + self.n_t))
        return probs


    def fit(self, iterations, debug=False):
        for iteration in range(iterations):
            # get probabilities for each group
            for d, doc in enumerate(self.data):
                for w, word_index in enumerate(doc):
                    # remove instance
                    self.rem_dwt(d, word_index, w)

                    # sample new topic
                    probs = self.sample_topics(d, word_index)

                    new_topic = numpy.random.multinomial(
                        1,
                        probs / probs.sum()).argmax()

                    # add document to new topic
                    self.add_dwt(d, word_index, w, new_topic)
            if debug:
                print "iteration {}, {}".format(
                    iteration,
                    Counter(self.dt.argmax(axis=1)))

    def rem_dwt(self, d, w, w_in_data):
        """
        Remove the counts for

        args:
            d - document
            w - word index
            t - topic
        """
        current_topic = self.dwt[d][w_in_data]
        self.dt[d, current_topic] -= 1
        self.tw[current_topic, w] -= 1
        self.n_t[current_topic] -= 1

    def add_dwt(self, d, w, w_in_data, t):
        """
        Remove the counts for

        args:
            d - document
            w - word index
            t - topic
        """
        self.dwt[d][w_in_data] = t
        self.dt[d, t] += 1
        self.tw[t, w] += 1
        self.n_t[t] += 1

#!/home/rob/.env/topics/bin/python
import random
from gensim import corpora
from ..helpers.exceptions import MissingModelError, MissingDataSetError
from ..wrappers import TNDMallet


class TND:
    topics = None

    def __init__(self, dataset=None, k=30, alpha=50, beta0=0.01, beta1=25, noise_words_max=200,
                 iterations=1000, top_words=20, topic_word_distribution=None,
                 noise_distribution=None, corpus=None, dictionary=None,
                 save_path=None, mallet_path=None, random_seed=1824, run=True,
                 workers=4):
        self.dataset = dataset
        self.k = k
        self.alpha = alpha
        self.beta0 = beta0
        self.beta1 = beta1
        self.noise_words_max = noise_words_max
        self.iterations = iterations
        self.top_words = top_words
        self.noise_distribution = noise_distribution
        self.topic_word_distribution = topic_word_distribution
        self.corpus = corpus
        self.dictionary = dictionary
        if save_path is not None:
            self.save_path = save_path
        else:
            self.save_path = 'tnd_results/'
        self.mallet_path = mallet_path
        self.random_seed = random_seed
        random.seed(self.random_seed)
        self.workers = workers

        if self.mallet_path is None and self.noise_distribution is None:
            raise MissingModelError('tnd')
        if self.dataset is None and (self.corpus is None or self.dictionary is None):
            raise MissingDataSetError

        if run:
            if (self.dataset is not None) and (self.corpus is None or self.dictionary is None):
                self.prepare_data()
            if self.noise_distribution is None:
                self.compute_tnd()

    def prepare_data(self):
        """
        takes dataset, sets self.dictionary and self.corpus for use in Mallet models and NLDA
        :return: void
        """
        dictionary = corpora.Dictionary(self.dataset)
        dictionary.filter_extremes()
        corpus = [dictionary.doc2bow(doc) for doc in self.dataset]
        self.dictionary = dictionary
        self.corpus = corpus

    def compute_tnd(self):
        """
        takes dataset, tnd parameters, tnd mallet path, and computes tnd model on dataset
        sets self.noise_distribution to the noise distribution computed in tnd
        :return: void
        """
        model = TNDMallet(self.mallet_path, self.corpus, num_topics=self.k, id2word=self.dictionary,
                          workers=self.workers,
                          alpha=self.alpha, beta=self.beta0, skew=self.beta1,
                          iterations=self.iterations, noise_words_max=self.noise_words_max,
                          random_seed=self.random_seed)
        noise = model.load_noise_dist()
        self.noise_distribution = noise
        self.topic_word_distribution = model.load_word_topics()
        topics = model.show_topics(num_topics=self.k, num_words=self.top_words, formatted=False)
        self.topics = [[w for (w, _) in topics[i][1]] for i in range(0, len(topics))]

    def get_topics(self, top_words=None):
        """
        takes top_words and self.topics, returns a list of topic lists of length top_words
        :param top_words:
        :return: list of topic lists
        """
        if top_words is None:
            top_words = self.top_words
        topics = self.topics
        if topics is None or len(topics) < 1:
            raise ValueError('No topics have been computed yet.')

        return [x[:top_words] for x in topics]

    def get_noise_distribution(self, tnd_noise_words_max=None):
        """
        takes self.noise_distribution and tnd_noise_words_max
        returns a list of (noise word, frequency) tuples ranked by frequency
        :param tnd_noise_words_max:
        :return: list of (noise word, frequency) tuples
        """
        if tnd_noise_words_max is None:
            tnd_noise_words_max = self.noise_words_max
        noise = self.noise_distribution
        if noise is None or len(noise) < 1:
            raise ValueError('No noise distribution has been computed yet.')

        noise_list = sorted([(x, int(noise[x])) for x in noise.keys()], key=lambda x: x[1], reverse=True)
        return noise_list[:tnd_noise_words_max]

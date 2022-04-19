import random
from gensim import corpora
from ..helpers.exceptions import MissingModelError, MissingDataSetError
from ..wrappers import TNDMallet


class TND:
    '''
    Topic-Noise Discriminator (TND).  The original Topic-Noise Model, this model is best used in an ensemble with other
    models, such as LDA (NLDA), or the Guided Topic Model (GTM).

    :param dataset: list of lists, required.
    :param k: int, optional:
        Number of topics to compute in TND.
    :param alpha: int, optional:
            Alpha parameter of TND.
    :param beta0: float, optional:
            Beta_0 parameter of TND.
    :param beta1: int, optional
            Beta_1 (skew) parameter of TND.
    :param noise_words_max: int, optional:
            Number of noise words to save when saving the distribution to a file.
            The top `noise_words_max` most probable noise words will be saved.
    :param iterations: int, optional:
            Number of training iterations for TND.
    :param top_words: int, optional:
        Number of words per topic to return.
    :param topic_word_distribution: dict, optional:
        Pre-trained topic-word distribution.
    :param noise_distribution: dict, optional:
        Pre-trained noise distribution.
    :param corpus: Gensim object, optional:
        Formatted documents for use in model.  Automatically computed if not provided.
    :param dictionary: Gensim object, optional:
        Formatted word mapping for use in model.  Automatically computed if not provided.
    :param mallet_path: path to Mallet TND code, required:
        Path should be `path/to/mallet-tnd/bin/mallet`.
    :param random_seed: int, optional:
        Seed for random-number generated processes.
    :param run: bool, optional:
        If true, run model on initialization, provided data is provided.
    :param workers: int, optional:
        Number of cores to use for computation of TND.
    '''

    def __init__(self, dataset=None, k=30, alpha=50, beta0=0.01, beta1=25, noise_words_max=200,
                 iterations=1000, top_words=20, topic_word_distribution=None,
                 noise_distribution=None, corpus=None, dictionary=None,
                 mallet_path=None, random_seed=1824, run=True,
                 workers=4):
        self.topics = []
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
                self._prepare_data()
            if self.noise_distribution is None:
                self._compute_tnd()

    def _prepare_data(self):
        """
        Takes dataset, sets self.dictionary and self.corpus for use in Mallet models and NLDA

        """
        dictionary = corpora.Dictionary(self.dataset)
        dictionary.filter_extremes()
        corpus = [dictionary.doc2bow(doc) for doc in self.dataset]
        self.dictionary = dictionary
        self.corpus = corpus

    def _compute_tnd(self):
        """
        Takes dataset, tnd parameters, tnd mallet path, and computes tnd model on dataset
        sets self.noise_distribution to the noise distribution computed in tnd


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
        Takes top_words and self.topics, returns a list of topic lists of length top_words

        :param top_words: number of words per topic
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

        :param tnd_noise_words_max: number of words to be returned
        :return: list of (noise word, frequency) tuples
        """
        if tnd_noise_words_max is None:
            tnd_noise_words_max = self.noise_words_max
        noise = self.noise_distribution
        if noise is None or len(noise) < 1:
            raise ValueError('No noise distribution has been computed yet.')

        noise_list = sorted([(x, int(noise[x])) for x in noise.keys()], key=lambda x: x[1], reverse=True)
        return noise_list[:tnd_noise_words_max]

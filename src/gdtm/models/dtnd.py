from gensim import corpora
from ..helpers.common import save_topics, save_noise_dist
from ..wrappers import dTNDMallet
from .tnd import TND


class dTND(TND):
    '''
    Dynamic Topic-Noise Discriminator (dTND).

    :param dataset: ordered list of sub-datasets,
        where dataset[i] is the data set for time period i
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
    :param save_path: filepath, optional:
        Path to save each time period's topics and noise distributions to
    :param starting_alpha_array_file: filepath, optional:
        Path of file containing initial alpha distributon.
    :param iterations: int, optional:
            Number of training iterations for TND.
    :param top_words: int, optional:
        Number of words per topic to return.
    :param tw_dist: dict, optional:
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
                 iterations=1000, top_words=20, topic_word_distribution=None, corpus=None, dictionary=None,
                 save_path=None, mallet_path=None, starting_alpha_array_file=None, random_seed=1824, run=True,
                 noise_distribution=None, workers=4):
        super().__init__(dataset=dataset, k=k, alpha=alpha, beta0=beta0, beta1=beta1, noise_words_max=noise_words_max,
                         iterations=iterations, top_words=top_words, topic_word_distribution=topic_word_distribution,
                         noise_distribution=noise_distribution, corpus=corpus, dictionary=dictionary,
                         mallet_path=mallet_path, random_seed=random_seed, run=False,
                         workers=workers)
        self.topics = None
        self.last_alpha_array_file = None
        self.last_beta = None
        self.last_noise_dist_file = None
        self.last_tw_dist_file = None
        if noise_distribution is None:
            self.noise_distribution = []

        if save_path is not None:
            save = True
            self.save_path = save_path
        else:
            save = False
            self.save_path = 'dtnd_results/'
        self.last_beta = self.beta0
        self.last_alpha_array_file = starting_alpha_array_file

        if run:
            self._run_all_time_periods(save=save)

    def _prepare_data(self, t):
        """
        takes dataset, sets self.dictionary and self.corpus for use in Mallet models and NLDA
        :return: void
        """
        dictionary = corpora.Dictionary(self.dataset[t])
        dictionary.filter_extremes()
        corpus = [dictionary.doc2bow(doc) for doc in self.dataset[t]]
        self.dictionary = dictionary
        self.corpus = corpus

    def _run_one_time_period(self, t):
        # pass previous noise and tw distribution files into mallet
        # prepare data for time period t
        self._prepare_data(t)
        # run model
        model = dTNDMallet(self.mallet_path, corpus=self.corpus, num_topics=self.k, beta=self.last_beta,
                           id2word=self.dictionary, iterations=self.iterations, skew=self.beta1,
                           noise_words_max=self.noise_words_max, workers=self.workers,
                           noise_dist_file=self.last_noise_dist_file, tw_dist_file=self.last_tw_dist_file,
                           alpha_array_infile=self.last_alpha_array_file, alpha=self.alpha)
        # get/set new beta and alpha array files, noise dist file
        self.last_beta = model.load_beta()
        self.last_alpha_array_file = model.falphaarrayfile()
        self.last_noise_dist_file = model.fnoisefile()
        self.last_tw_dist_file = model.fwordweights()
        return model

    def _run_all_time_periods(self, save=True):
        for t in range(0, len(self.dataset)):
            model = self._run_one_time_period(t)
            topics = model.show_topics(num_topics=self.k, num_words=self.top_words, formatted=False)
            noise = model.load_noise_dist()
            self.topics = topics
            self.noise_distribution.append(noise)
            if save:
                topics = model.show_topics(num_topics=self.k, num_words=self.top_words, formatted=False)
                topics = [[w for (w, _) in topic[1]] for topic in topics]
                save_topics(topics, self.save_path + 'topics_{}_{}.csv'.format(self.k, t))
                noise_list = sorted([(x, noise[x]) for x in noise.keys()], key=lambda x: x[1], reverse=True)
                save_noise_dist(noise_list, self.save_path + 'noise_{}_{}.csv'.format(self.k, t))

    def get_topics(self, t, top_words=None):
        """
        takes top_words and self.topics, returns a list of topic lists of length top_words

        :param t: time period
        :param top_words:
        :return: list of topic lists
        """
        if top_words is None:
            top_words = self.top_words
        topics = self.topics
        if len(topics) < 1:
            raise ValueError('No topics have been computed yet.')
        elif len(topics[t]) < 1:
            raise ValueError('No topics have been computed for this time period yet.')

        return [x[:top_words] for x in topics[t]]

    def get_noise_distribution(self, t, noise_words_max=None):
        """
        takes self.tnd_noise_distribution and tnd_noise_words_max
        returns a list of (noise word, frequency) tuples ranked by frequency

        :param t: time period
        :param tnd_noise_words_max:
        :return: list of (noise word, frequency) tuples
        """
        if noise_words_max is None:
            noise_words_max = self.noise_words_max
        if len(self.noise_distribution) < 1:
            raise ValueError('No noise distribution has been computed yet.')
        elif len(self.noise_distribution[t]) < 1:
            raise ValueError('No noise distribution has been computed for this time period yet.')

        noise = self.noise_distribution[t]
        noise_list = sorted([(x, int(noise[x])) for x in noise.keys()], key=lambda x: x[1], reverse=True)
        return noise_list[:noise_words_max]

    def __str__(self):
        return 'dTND'

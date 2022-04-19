from ..helpers.common import save_topics, save_noise_dist
from ..wrappers import deTNDMallet
from .dtnd import dTND


class deTND(dTND):
    '''
    Dynamic Embedded Topic-Noise Discriminator (deTND).

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
    :param embedding_path: filepath, required:
        Path to trained word embedding vectors.
    :param closest_x_words: int, optional:
        The number of words to sample from the word embedding space each time a word is determined to be a noise word.
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
                 iterations=1000, top_words=20, tw_dist=None, corpus=None, dictionary=None,
                 save_path=None, mallet_path=None, starting_alpha_array_file=None, embedding_path=None,
                 closest_x_words=3, noise_distribution=None,
                 random_seed=1824, run=True,
                 workers=4):

        super().__init__(dataset=dataset, k=k, alpha=alpha, beta0=beta0, beta1=beta1, noise_words_max=noise_words_max,
                         iterations=iterations, top_words=top_words, topic_word_distribution=tw_dist, corpus=corpus,
                         dictionary=dictionary, mallet_path=mallet_path,
                         starting_alpha_array_file=starting_alpha_array_file, random_seed=random_seed,
                         noise_distribution=noise_distribution,
                         run=False, workers=workers)

        if save_path is not None:
            save = True
            self.save_path = save_path
        else:
            save = False
            self.save_path = 'detnd_results/'
        self.embedding_path = embedding_path
        self.closest_x_words = closest_x_words

        if run:
            self._run_all_time_periods(save=save)

    def _run_one_time_period(self, t):
        # pass previous noise and tw distribution files into mallet
        # prepare data for time period t
        self._prepare_data(t)
        # run model
        if self.last_alpha_array_file is not None:
            model = deTNDMallet(self.mallet_path, corpus=self.corpus, num_topics=self.k, beta=self.last_beta,
                                id2word=self.dictionary, iterations=self.iterations, skew=self.beta1,
                                noise_words_max=self.noise_words_max, workers=self.workers,
                                noise_dist_file=self.last_noise_dist_file, tw_dist_file=self.last_tw_dist_file,
                                alpha_array_infile=self.last_alpha_array_file, embedding_path=self.embedding_path,
                                closest_x_words=self.closest_x_words)
        else:
            model = deTNDMallet(self.mallet_path, corpus=self.corpus, num_topics=self.k, beta=self.last_beta,
                                id2word=self.dictionary, iterations=self.iterations, skew=self.beta1,
                                noise_words_max=self.noise_words_max, workers=self.workers,
                                alpha=self.alpha, embedding_path=self.embedding_path,
                                closest_x_words=self.closest_x_words)
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
            self.noise_distribution = noise
            if save:
                topics = model.show_topics(num_topics=self.k, num_words=self.top_words, formatted=False)
                topics = [[w for (w, _) in topic[1]] for topic in topics]
                save_topics(topics, self.save_path + 'topics_{}_{}_{}.csv'.format(self.k, self.closest_x_words, t))
                noise_list = sorted([(x, noise[x]) for x in self.noise_distribution.keys()], key=lambda x: x[1], reverse=True)
                save_noise_dist(noise_list, self.save_path + 'noise_{}_{}_{}.csv'.format(self.k, self.closest_x_words, t))

    def __str__(self):
        return 'deTND'

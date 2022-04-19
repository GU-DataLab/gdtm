from ..helpers.exceptions import MissingEmbeddingPathError
from ..wrappers import eTNDMallet
from .tnd import TND


class eTND(TND):
    '''
    Embedded Topic-Noise Discriminator (eTND).
    The embedded version of TND, this model is best used in an ensemble with other
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
    :param embedding_path: filepath, required:
        path to trained word embedding vectors.
    :param closest_x_words: int, optional:
        The number of words to sample from the word embedding space each time a word is determined to be a noise word.
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
                 mallet_path=None, embedding_path=None,
                 closest_x_words=3, random_seed=1824, run=True, workers=4):
        super().__init__(dataset=dataset, k=k, alpha=alpha, beta0=beta0, beta1=beta1, noise_words_max=noise_words_max,
                         iterations=iterations, top_words=top_words, topic_word_distribution=topic_word_distribution,
                         noise_distribution=noise_distribution, corpus=corpus, dictionary=dictionary,
                         mallet_path=mallet_path, random_seed=random_seed, run=False, workers=workers)
        self.topics = []
        self.embedding_path = embedding_path
        if self.embedding_path is None:
            raise MissingEmbeddingPathError
        self.closest_x_words = closest_x_words

        if run:
            if (self.dataset is not None) and (self.corpus is None or self.dictionary is None):
                self._prepare_data()
            if self.noise_distribution is None:
                self._compute_tnd()

    def _compute_tnd(self):
        """
        takes dataset, tnd parameters, tnd mallet path, and computes tnd model on dataset
        sets self.noise_distribution to the noise distribution computed in tnd
        :return: void
        """
        model = eTNDMallet(self.mallet_path, self.corpus, num_topics=self.k, id2word=self.dictionary,
                           workers=self.workers, alpha=self.alpha, beta=self.beta0, skew=self.beta1,
                           iterations=self.iterations, noise_words_max=self.noise_words_max,
                           random_seed=self.random_seed, closest_x_words=self.closest_x_words,
                           embedding_path=self.embedding_path)
        noise = model.load_noise_dist()
        self.noise_distribution = noise
        self.topic_word_distribution = model.load_word_topics()
        topics = model.show_topics(num_topics=self.k, num_words=self.top_words, formatted=False)
        self.topics = [[w for (w, _) in topics[i][1]] for i in range(0, len(topics))]

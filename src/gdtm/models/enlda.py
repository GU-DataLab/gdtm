from ..helpers.exceptions import MissingEmbeddingPathError
from ..wrappers import eTNDMallet
from .nlda import NLDA


class eNLDA(NLDA):
    '''
    Embedded Noiseless Latent Dirichlet Allocation (eNLDA).
    An ensemble topic-noise model consisting of the noise distribution from
    TND and the topic-word distribution from LDA.
    Input the raw data and compute the whole model, or input pre-computed distributions
    for faster inference.
    Uses word embedding vectors to enhance the TND noise distribution.

    :param dataset: list of lists, required.
    :param tnd_k: int, optional:
        Number of topics to compute in TND.
    :param tnd_alpha: int, optional:
            Alpha parameter of TND.
    :param tnd_beta0: float, optional:
            Beta_0 parameter of TND.
    :param tnd_beta1: int, optional
            Beta_1 (skew) parameter of TND.
    :param tnd_noise_words_max: int, optional:
            Number of noise words to save when saving the distribution to a file.
            The top `noise_words_max` most probable noise words will be saved.
    :param tnd_iterations: int, optional:
            Number of training iterations for TND.
    :param lda_iterations: int, optional:
            Number of training iterations for LDA.
    :param lda_k: int, optional:
        Number of topics to compute in LDA.
    :param phi: int, optional:
        Topic weighting for noise filtering step.
    :param topic_depth: int, optional:
        Number of most probable words per topic to consider for replacement in noise filtering step.
    :param embedding_path: filepath, required:
        Path to trained word embedding vectors.
    :param closest_x_words: int, optional:
        The number of words to sample from the word embedding space each time a word is determined to be a noise word.
    :param top_words: int, optional:
        Number of words per topic to return.
    :param tnd_noise_distribution: dict, optional:
        Pre-trained noise distribution
    :param lda_tw_dist: dict, optional:
        Pre-trained topic-word distribution.
    :param lda_topics: list of lists, optional:
        Pre-computed LDA topics.
    :param corpus: Gensim object, optional:
        Formatted documents for use in model.  Automatically computed if not provided.
    :param dictionary: Gensim object, optional:
        Formatted word mapping for use in model.  Automatically computed if not provided.
    :param mallet_tnd_path: path to Mallet TND code, required:
        Path should be `path/to/mallet-tnd/bin/mallet`.
    :param mallet_lda_path: path to Mallet LDA code, required:
        Path should be `path/to/mallet-lda/bin/mallet`.
    :param random_seed: int, optional:
        Seed for random-number generated processes.
    :param run: bool, optional:
        If true, run model on initialization, if data is provided.
    :param tnd_workers: int, optional:
        Number of cores to use for computation of TND.
    :param lda_workers: int, optional:
        Number of cores to use for computation of LDA.
    '''

    def __init__(self, dataset=None, tnd_k=30, tnd_alpha=50, tnd_beta0=0.01, tnd_beta1=25, tnd_noise_words_max=200,
                 tnd_iterations=1000, lda_iterations=1000, lda_k=30, phi=10, topic_depth=100, top_words=20,
                 tnd_noise_distribution=None, lda_tw_dist=None, lda_topics=None, corpus=None, dictionary=None,
                 mallet_tnd_path=None, mallet_lda_path=None, embedding_path=None,
                 closest_x_words=3, random_seed=1824, run=True, tnd_workers=4, lda_workers=4):
        super().__init__(dataset=dataset, tnd_k=tnd_k, tnd_alpha=tnd_alpha, tnd_beta0=tnd_beta0, tnd_beta1=tnd_beta1,
                         tnd_noise_words_max=tnd_noise_words_max, tnd_iterations=tnd_iterations,
                         lda_iterations=lda_iterations, lda_k=lda_k, phi=phi,
                         topic_depth=topic_depth, top_words=top_words,
                         tnd_noise_distribution=tnd_noise_distribution, lda_tw_dist=lda_tw_dist, lda_topics=lda_topics,
                         corpus=corpus, dictionary=dictionary, mallet_tnd_path=mallet_tnd_path,
                         mallet_lda_path=mallet_lda_path, random_seed=random_seed, run=False, tnd_workers=tnd_workers,
                         lda_workers=lda_workers)
        self.mallet_tnd_path = mallet_tnd_path
        self.mallet_lda_path = mallet_lda_path
        self.embedding_path = embedding_path
        if self.embedding_path is None:
            raise MissingEmbeddingPathError
        self.closest_x_words = closest_x_words

        if run:
            if (self.dataset is not None) and (self.corpus is None or self.dictionary is None):
                self._prepare_data()
            if self.tnd_noise_distribution is None:
                self._compute_tnd()
            if self.lda_tw_dist is None:
                self._compute_lda()

            self._compute_nlda()

    def _compute_tnd(self):
        """
        takes dataset, tnd parameters, tnd mallet path, and computes tnd model on dataset
        sets self.tnd_noise_distribution to the noise distribution computed in tnd
        :return: void
        """
        model = eTNDMallet(self.mallet_tnd_path, self.corpus, num_topics=self.tnd_k, id2word=self.dictionary,
                           workers=self.tnd_workers, alpha=self.tnd_alpha, beta=self.tnd_beta0, skew=self.tnd_beta1,
                           iterations=self.tnd_iterations, noise_words_max=self.tnd_noise_words_max,
                           random_seed=self.random_seed, closest_x_words=self.closest_x_words,
                           embedding_path=self.embedding_path)
        noise = model.load_noise_dist()
        self.tnd_noise_distribution = noise

#!/home/rob/.env/topics/bin/python
import math
import random
from gensim import corpora
from ..helpers.exceptions import MissingModelError, MissingDataSetError, MissingEmbeddingPathError
from ..wrappers import eTNDMallet
from .nlda import NLDA


class eNLDA(NLDA):
    topics = None

    def __init__(self, dataset=None, tnd_k=30, tnd_alpha=50, tnd_beta0=0.01, tnd_beta1=25, tnd_noise_words_max=200,
                 tnd_iterations=1000, lda_iterations=1000, lda_k=30, nlda_phi=10, nlda_topic_depth=100, top_words=20,
                 tnd_noise_distribution=None, lda_tw_dist=None, lda_topics=None, corpus=None, dictionary=None,
                 save_path=None, mallet_tnd_path=None, mallet_lda_path=None, embedding_path=None,
                 closest_x_words=3, random_seed=1824, run=True, tnd_workers=4, lda_workers=4):

        super().__init__(dataset=dataset, tnd_k=tnd_k, tnd_alpha=tnd_alpha, tnd_beta0=tnd_beta0, tnd_beta1=tnd_beta1,
                         tnd_noise_words_max=tnd_noise_words_max, tnd_iterations=tnd_iterations,
                         lda_iterations=lda_iterations, lda_k=lda_k, nlda_phi=nlda_phi,
                         nlda_topic_depth=nlda_topic_depth, top_words=top_words,
                         tnd_noise_distribution=tnd_noise_distribution, lda_tw_dist=lda_tw_dist, lda_topics=lda_topics,
                         corpus=corpus, dictionary=dictionary, save_path=save_path, mallet_tnd_path=mallet_tnd_path,
                         mallet_lda_path=mallet_lda_path, random_seed=random_seed, run=False, tnd_workers=tnd_workers,
                         lda_workers=lda_workers)

        if save_path is not None:
            self.save_path = save_path
        else:
            self.save_path = 'nlda_results/'
        self.mallet_tnd_path = mallet_tnd_path
        self.mallet_lda_path = mallet_lda_path
        self.embedding_path = embedding_path
        if self.embedding_path is None:
            raise MissingEmbeddingPathError
        self.closest_x_words = closest_x_words

        if run:
            if (self.dataset is not None) and (self.corpus is None or self.dictionary is None):
                self.prepare_data()
            if self.tnd_noise_distribution is None:
                self.compute_tnd()
            if self.lda_tw_dist is None:
                self.compute_lda()

            self.compute_nlda()

    def compute_tnd(self):
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

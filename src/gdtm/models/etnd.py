#!/home/rob/.env/topics/bin/python
from ..helpers.exceptions import MissingEmbeddingPathError
from ..wrappers import eTNDMallet
from .tnd import TND


class eTND(TND):
    """

    """
    topics = None

    def __init__(self, dataset=None, k=30, alpha=50, beta0=0.01, beta1=25, noise_words_max=200,
                 iterations=1000, top_words=20, topic_word_distribution=None,
                 noise_distribution=None, corpus=None, dictionary=None,
                 save_path=None, mallet_path=None, embedding_path=None,
                 closest_x_words=3, random_seed=1824, run=True, workers=4):

        super().__init__(dataset=dataset, k=k, alpha=alpha, beta0=beta0, beta1=beta1, noise_words_max=noise_words_max,
                         iterations=iterations, top_words=top_words, topic_word_distribution=topic_word_distribution,
                         noise_distribution=noise_distribution, corpus=corpus, dictionary=dictionary,
                         save_path=save_path, mallet_path=mallet_path, random_seed=random_seed, run=False,
                         workers=workers)

        if save_path is not None:
            self.save_path = save_path
        else:
            self.save_path = 'etnd_results/'
        self.embedding_path = embedding_path
        if self.embedding_path is None:
            raise MissingEmbeddingPathError
        self.closest_x_words = closest_x_words

        if run:
            if (self.dataset is not None) and (self.corpus is None or self.dictionary is None):
                self.prepare_data()
            if self.noise_distribution is None:
                self.compute_tnd()

    def compute_tnd(self):
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

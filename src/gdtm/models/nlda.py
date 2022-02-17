#!/home/rob/.env/topics/bin/python
import math
import random
from gensim import corpora
from ..helpers.exceptions import MissingModelError, MissingDataSetError
from ..wrappers import LDAMallet, TNDMallet


class NLDA:
    topics = None

    def __init__(self, dataset=None, tnd_k=30, tnd_alpha=50, tnd_beta0=0.01, tnd_beta1=25, tnd_noise_words_max=200,
                 tnd_iterations=1000, lda_iterations=1000, lda_k=30, nlda_phi=10, nlda_topic_depth=100, top_words=20,
                 tnd_noise_distribution=None, lda_tw_dist=None, lda_topics=None, corpus=None, dictionary=None,
                 save_path=None, mallet_tnd_path=None, mallet_lda_path=None, random_seed=1824, run=True,
                 tnd_workers=4, lda_workers=4):
        self.dataset = dataset
        self.tnd_k = tnd_k
        self.tnd_alpha = tnd_alpha
        self.tnd_beta0 = tnd_beta0
        self.tnd_beta1 = tnd_beta1
        self.tnd_noise_words_max = tnd_noise_words_max
        self.tnd_iterations = tnd_iterations
        self.lda_iterations = lda_iterations
        self.lda_k = lda_k
        self.nlda_phi = nlda_phi
        self.nlda_topic_depth = nlda_topic_depth
        self.top_words = top_words
        self.tnd_noise_distribution = tnd_noise_distribution
        self.lda_tw_dist = lda_tw_dist
        self.lda_topics = lda_topics
        self.corpus = corpus
        self.dictionary = dictionary
        if save_path is not None:
            self.save_path = save_path
        else:
            self.save_path = 'nlda_results/'
        self.mallet_tnd_path = mallet_tnd_path
        self.mallet_lda_path = mallet_lda_path
        self.random_seed = random_seed
        random.seed(self.random_seed)
        self.lda_workers = lda_workers
        self.tnd_workers = tnd_workers

        if self.mallet_tnd_path is None and self.tnd_noise_distribution is None:
            raise MissingModelError('tnd')
        if self.mallet_lda_path is None and self.lda_tw_dist is None:
            raise MissingModelError('lda')
        if self.dataset is None and (self.corpus is None or self.dictionary is None
                                     or self.tnd_noise_distribution is None or self.lda_tw_dist is None):
            raise MissingDataSetError

        if run:
            if (self.dataset is not None) and (self.corpus is None or self.dictionary is None):
                self.prepare_data()
            if self.tnd_noise_distribution is None:
                self.compute_tnd()
            if self.lda_tw_dist is None:
                self.compute_lda()

            self.compute_nlda()

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
        sets self.tnd_noise_distribution to the noise distribution computed in tnd
        :return: void
        """
        model = TNDMallet(self.mallet_tnd_path, self.corpus, num_topics=self.tnd_k, id2word=self.dictionary,
                          workers=self.tnd_workers,
                          alpha=self.tnd_alpha, beta=self.tnd_beta0, skew=self.tnd_beta1,
                          iterations=self.tnd_iterations, noise_words_max=self.tnd_noise_words_max,
                          random_seed=self.random_seed)
        noise = model.load_noise_dist()
        self.tnd_noise_distribution = noise

    def compute_lda(self):
        """
        takes dataset, lda parameters, lda mallet path, and computes LDA model on dataset
        sets self.lda_tw_dist to the topic word distribution computed in LDA
        :return: void
        """
        model = LDAMallet(self.mallet_lda_path, self.corpus, num_topics=self.lda_k, id2word=self.dictionary,
                          workers=self.lda_workers, iterations=self.lda_iterations, random_seed=self.random_seed)
        topic_word_distribution = model.load_word_topics()
        self.lda_tw_dist = topic_word_distribution
        topics = model.show_topics(num_topics=self.lda_k, num_words=self.nlda_topic_depth, formatted=False)
        self.lda_topics = [[w for (w, _) in topics[i][1]] for i in range(0, len(topics))]

    def compute_nlda(self):
        """
        takes self.tnd_noise_distribution, self.lda_tw_dist, self.phi, self.top_words, and computes NLDA topics
        sets self.topics to the set of topics computed from noise distribution and topic word distribution
        :return: void
        """
        topics = []
        for i in range(0, len(self.lda_topics)):
            topic = self.lda_topics[i]
            final_topic = []
            j = 0
            while len(topic) > j and len(final_topic) < self.top_words and j < self.nlda_topic_depth:
                w = topic[j]
                token_id = self.dictionary.token2id[w]
                beta = 2
                if w in self.tnd_noise_distribution:
                    beta += self.tnd_noise_distribution[w]
                beta = max(2, beta * (self.nlda_phi / self.lda_k))
                alpha = 2 + self.lda_tw_dist[i, token_id]
                roll = random.betavariate(alpha=math.sqrt(alpha), beta=math.sqrt(beta))
                if roll >= 0.5:
                    final_topic.append(w)
                    if w not in self.tnd_noise_distribution:
                        self.tnd_noise_distribution[w] = 0
                    self.tnd_noise_distribution[w] += (alpha - 2)
                j += 1
            topics.append(final_topic)
        self.topics = topics

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
        takes self.tnd_noise_distribution and tnd_noise_words_max
        returns a list of (noise word, frequency) tuples ranked by frequency
        :param tnd_noise_words_max:
        :return: list of (noise word, frequency) tuples
        """
        if tnd_noise_words_max is None:
            tnd_noise_words_max = self.tnd_noise_words_max
        noise = self.tnd_noise_distribution
        if noise is None or len(noise) < 1:
            raise ValueError('No noise distribution has been computed yet.')

        noise_list = sorted([(x, int(noise[x])) for x in noise.keys()], key=lambda x: x[1], reverse=True)
        return noise_list[:tnd_noise_words_max]

#!/home/rob/.env/topics/bin/python
import math
import random
from ..wrappers import dLDAMallet, deTNDMallet
from .dnlda import dNLDA
from ..helpers.common import save_topics


class deNLDA(dNLDA):
    topics = []

    def __init__(self, dataset=None, k=30, tnd_alpha=50, tnd_beta0=0.01, tnd_beta1=25, tnd_noise_words_max=200,
                 tnd_iterations=1000, lda_iterations=1000, lda_alpha=50, lda_beta=0.01, nlda_phi=10,
                 nlda_topic_depth=100, top_words=20, nlda_phi_list=None, nlda_topic_depth_list=None,
                 num_time_periods=10, tnd_noise_distribution=[], lda_tw_dist=[], lda_topics=[], corpus=None,
                 dictionary=None, last_tnd_alpha_array_file=None, last_tnd_noise_dist_file=None,
                 last_tnd_tw_dist_file=None, last_lda_alpha_array_file=None, last_lda_tw_dist_file=None,
                 lda_save_path=None, save_path=None, mallet_tnd_path=None, mallet_lda_path=None, embedding_path=None,
                 closest_x_words=3, random_seed=1824, workers=4, run=True):

        super().__init__(dataset, k, tnd_alpha, tnd_beta0, tnd_beta1, tnd_noise_words_max, tnd_iterations, lda_iterations,
                       lda_alpha, lda_beta, nlda_phi, nlda_topic_depth, top_words, nlda_phi_list, nlda_topic_depth_list,
                       num_time_periods, tnd_noise_distribution, lda_tw_dist, lda_topics, corpus, dictionary,
                       last_tnd_alpha_array_file, last_tnd_noise_dist_file, last_tnd_tw_dist_file, last_lda_alpha_array_file,
                       last_lda_tw_dist_file, lda_save_path, save_path, mallet_tnd_path, mallet_lda_path, random_seed,
                       workers, run=False)
        if save_path is not None:
            save = True
            self.save_path = save_path
        else:
            save = False
            self.save_path = 'denlda_results/'
        self.embedding_path = embedding_path
        self.closest_x_words = closest_x_words

        if run:
            self.run_nlda_all_time_periods(save=save)

    def compute_nlda(self, noise_dist, tw_dist, lda_topics, phi, depth):
        """
        takes self.tnd_noise_distribution, self.lda_tw_dist, self.phi, self.top_words, and computes NLDA topics
        sets self.topics to the set of topics computed from noise distribution and topic word distribution
        :return: void
        """
        topics = []
        for i in range(0, len(lda_topics)):
            topic = lda_topics[i]
            final_topic = []
            j = 0
            while len(topic) > j and len(final_topic) < self.top_words and j < depth:
                w = topic[j]
                token_id = self.dictionary.token2id[w]
                beta = 2
                if w in noise_dist:
                    beta += noise_dist[w]
                beta = max(2, beta * (phi / self.k))
                alpha = 2 + tw_dist[i, token_id]
                roll = random.betavariate(alpha=math.sqrt(alpha), beta=math.sqrt(beta))
                if roll >= 0.5:
                    final_topic.append(w)
                    if w not in noise_dist:
                        noise_dist[w] = 0
                    noise_dist[w] += (alpha - 2)
                j += 1
            topics.append(final_topic)
        return topics

    def run_nlda_one_time_period(self, t):
        self.prepare_data(t)
        if len(self.tnd_noise_distribution) > t:
            noise_dist = self.tnd_noise_distribution[t]
        else:
            tnd_model = deTNDMallet(self.mallet_tnd_path, corpus=self.corpus, num_topics=self.k, beta=self.last_tnd_beta,
                                    id2word=self.dictionary, iterations=self.tnd_iterations, skew=self.tnd_beta1,
                                    noise_words_max=self.tnd_noise_words_max, workers=self.tnd_workers,
                                    noise_dist_file=self.last_tnd_noise_dist_file,
                                    tw_dist_file=self.last_tnd_tw_dist_file,
                                    alpha_array_infile=self.last_tnd_alpha_array_file, alpha=self.tnd_alpha,
                                    embedding_path=self.embedding_path, closest_x_words=self.closest_x_words)
            noise_dist = tnd_model.load_noise_dist()
            self.tnd_noise_distribution.append(noise_dist)
            self.last_tnd_beta = tnd_model.load_beta()
            self.last_tnd_alpha_array_file = tnd_model.falphaarrayfile()
            self.last_tnd_noise_dist_file = tnd_model.fnoisefile()
            self.last_tnd_tw_dist_file = tnd_model.fwordweights()

        if len(self.lda_tw_dist) > t and len(self.lda_topics) > t:
            tw_dist = self.lda_tw_dist[t]
            lda_topics = self.lda_topics[t]
        else:
            lda_model = dLDAMallet(self.mallet_lda_path, self.corpus, num_topics=self.k, id2word=self.dictionary,
                               workers=self.lda_workers, tw_dist_file=self.last_lda_tw_dist_file,
                               alpha_array_infile=self.last_lda_alpha_array_file, beta=self.last_lda_beta,
                               iterations=self.lda_iterations, random_seed=self.random_seed)
            tw_dist = lda_model.load_word_topics()
            self.lda_tw_dist.append(tw_dist)
            topic_tuples = lda_model.show_topics(num_topics=self.lda_k, num_words=self.nlda_topic_depth, formatted=False)
            lda_topics = [[w for (w, _) in topic[1]] for topic in topic_tuples]
            if self.lda_save_path:
                save_topics(lda_topics, self.lda_save_path + 'topics_{}_{}.csv'.format(self.k, t))
            self.lda_topics.append(lda_topics)
            self.last_lda_beta = lda_model.load_beta()
            self.last_lda_alpha_array_file = lda_model.falphaarrayfile()
            self.last_lda_tw_dist_file = lda_model.fwordweights()

        topic_sets = self.test_all_phi_and_depths(tw_dist, noise_dist, lda_topics)
        return topic_sets, noise_dist

    def __str__(self):
        return 'deNLDA'

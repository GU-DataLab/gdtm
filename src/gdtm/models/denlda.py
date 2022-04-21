import math
import random
from ..wrappers import dLDAMallet, deTNDMallet
from .dnlda import dNLDA
from ..helpers.common import save_topics


class deNLDA(dNLDA):
    '''
    Dynamic Embedded Noiseless Latent Dirichlet Allocation (deNLDA).


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
    :param lda_beta: float, optional:
        Initial Beta parameter for d-LDA.
    :param phi: int, optional:
        Topic weighting for noise filtering step.
    :param topic_depth: int, optional:
        Number of most probable words per topic to consider for replacement in noise filtering step.
    :param top_words: int, optional:
        Number of words per topic to return.
    :param phi_list: list, optional:
        List of phi parameters to test on.
    :param topic_depth_list: list, optional:
        List of topic_depth parameters to test on.
    :param num_time_periods: int, optional:
        Number of time periods in the data set.
        Should be equal to the length of the data set parameter.
    :param embedding_path: filepath, required:
        Path to trained word embedding vectors.
    :param closest_x_words: int, optional:
        The number of words to sample from the word embedding space each time a word is determined to be a noise word.
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
    :param last_tnd_alpha_array_file: filepath, optional:
        Path to file containing previous time period's d-TND alpha array.
    :param last_tnd_noise_dist_file: filepath, optional:
        Path to file containing previous time period's noise distribution.
    :param last_tnd_tw_dist_file: filepath, optional:
        Path to file containing previous time period's d-TND topic-word distribution.
    :param last_lda_alpha_array_file: filepath, optional:
        Path to file containing previous time period's d-LDA alpha array.
    :param last_lda_tw_dist_file: filepath, optional:
        Path to file containing previous time period's d-LDA topic-word distribution.
    :param lda_save_path: filepath, optional:
        Path to save d-LDA topics for each time period. Each time period is saved in a separate file in this directory.
    :param save_path: filepath, optional:
        Path to save DNLDA topics for each time period. Each time period is saved in a separate file in this directory.
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
                 tnd_iterations=1000, lda_iterations=1000, lda_k=30, lda_beta=0.01, phi=10,
                 topic_depth=100, top_words=20, phi_list=None, topic_depth_list=None,
                 num_time_periods=10, tnd_noise_distribution=[], lda_tw_dist=[], lda_topics=[], corpus=None,
                 dictionary=None, last_tnd_alpha_array_file=None, last_tnd_noise_dist_file=None,
                 last_tnd_tw_dist_file=None, last_lda_alpha_array_file=None, last_lda_tw_dist_file=None,
                 lda_save_path=None, save_path=None, mallet_tnd_path=None, mallet_lda_path=None, embedding_path=None,
                 closest_x_words=3, random_seed=1824, lda_workers=4, tnd_workers=4, run=True):
        super().__init__(dataset=dataset, tnd_k=tnd_k, tnd_alpha=tnd_alpha, tnd_beta0=tnd_beta0, tnd_beta1=tnd_beta1,
                         tnd_noise_words_max=tnd_noise_words_max, tnd_iterations=tnd_iterations, lda_k=lda_k,
                         lda_iterations=lda_iterations, lda_beta=lda_beta, phi=phi, topic_depth=topic_depth,
                         top_words=top_words, phi_list=phi_list, topic_depth_list=topic_depth_list,
                         num_time_periods=num_time_periods, tnd_noise_distribution=tnd_noise_distribution,
                         lda_tw_dist=lda_tw_dist, lda_topics=lda_topics, corpus=corpus, dictionary=dictionary,
                         last_tnd_alpha_array_file=last_tnd_alpha_array_file,
                         last_tnd_noise_dist_file=last_tnd_noise_dist_file, last_tnd_tw_dist_file=last_tnd_tw_dist_file,
                         last_lda_alpha_array_file=last_lda_alpha_array_file,
                         last_lda_tw_dist_file=last_lda_tw_dist_file, lda_save_path=lda_save_path, save_path=save_path,
                         mallet_tnd_path=mallet_tnd_path, mallet_lda_path=mallet_lda_path, random_seed=random_seed,
                         lda_workers=lda_workers, tnd_workers=tnd_workers, run=False)
        if save_path is not None:
            save = True
            self.save_path = save_path
        else:
            save = False
            self.save_path = 'denlda_results/'
        self.embedding_path = embedding_path
        self.closest_x_words = closest_x_words

        if run:
            self._run_nlda_all_time_periods(save=save)

    def _compute_nlda(self, noise_dist, tw_dist, lda_topics, phi, depth):
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
                beta = max(2, beta * (phi / self.lda_k))
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

    def _run_nlda_one_time_period(self, t):
        self._prepare_data(t)
        if len(self.tnd_noise_distribution) > t:
            noise_dist = self.tnd_noise_distribution[t]
        else:
            tnd_model = deTNDMallet(self.mallet_tnd_path, corpus=self.corpus, num_topics=self.tnd_k,
                                    beta=self.last_tnd_beta, id2word=self.dictionary, iterations=self.tnd_iterations,
                                    skew=self.tnd_beta1, noise_words_max=self.tnd_noise_words_max,
                                    workers=self.tnd_workers, noise_dist_file=self.last_tnd_noise_dist_file,
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
            lda_model = dLDAMallet(self.mallet_lda_path, self.corpus, num_topics=self.lda_k, id2word=self.dictionary,
                               workers=self.lda_workers, tw_dist_file=self.last_lda_tw_dist_file,
                               alpha_array_infile=self.last_lda_alpha_array_file, beta=self.last_lda_beta,
                               iterations=self.lda_iterations, random_seed=self.random_seed)
            tw_dist = lda_model.load_word_topics()
            self.lda_tw_dist.append(tw_dist)
            topic_tuples = lda_model.show_topics(num_topics=self.lda_k, num_words=self.nlda_topic_depth,
                                                 formatted=False)
            lda_topics = [[w for (w, _) in topic[1]] for topic in topic_tuples]
            if self.lda_save_path:
                save_topics(lda_topics, self.lda_save_path + 'topics_{}_{}.csv'.format(self.lda_k, t))
            self.lda_topics.append(lda_topics)
            self.last_lda_beta = lda_model.load_beta()
            self.last_lda_alpha_array_file = lda_model.falphaarrayfile()
            self.last_lda_tw_dist_file = lda_model.fwordweights()

        topic_sets = self._test_all_phi_and_depths(tw_dist, noise_dist, lda_topics)
        return topic_sets, noise_dist

    def __str__(self):
        return 'deNLDA'

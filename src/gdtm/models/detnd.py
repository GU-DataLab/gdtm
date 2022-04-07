#!/home/rob/.env/topics/bin/python
from ..helpers.common import save_topics, save_noise_dist
from ..wrappers import deTNDMallet
from .dtnd import dTND


class deTND(dTND):
    topics = None
    last_alpha_array_file = None
    last_beta = None
    last_noise_dist_file = None
    last_tw_dist_file = None

    def __init__(self, dataset=None, k=30, alpha=50, beta0=0.01, beta1=25, noise_words_max=200,
                 iterations=1000, top_words=20, tw_dist=None, corpus=None, dictionary=None,
                 save_path=None, mallet_tnd_path=None, starting_alpha_array_file=None, embedding_path=None,
                 closest_x_words=3,
                 random_seed=1824, run=True,
                 workers=4):
        '''

        :param dataset: dataset should be an ordered list of sub-datasets,
                        where dataset[i] is the data set for time period i
        :param k:
        :param alpha:
        :param beta0:
        :param beta1:
        :param noise_words_max:
        :param iterations:
        :param top_words:
        :param tw_dist:
        :param corpus:
        :param dictionary:
        :param save_path:
        :param mallet_tnd_path:
        :param starting_beta_file:
        :param starting_alpha_array_file:
        :param random_seed:
        :param run:
        '''

        super().__init__(dataset, k, alpha, beta0, beta1, noise_words_max, iterations,
                         top_words, tw_dist, corpus, dictionary, save_path, mallet_tnd_path,
                         starting_alpha_array_file, random_seed, run=False, workers=workers)
        if save_path is not None:
            save = True
            self.save_path = save_path
        else:
            save = False
            self.save_path = 'detnd_results/'
        self.embedding_path = embedding_path
        self.closest_x_words = closest_x_words

        if run:
            self.run_all_time_periods(save=save)

    def run_one_time_period(self, t):
        # pass previous noise and tw distribution files into mallet
        # prepare data for time period t
        self.prepare_data(t)
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

    def run_all_time_periods(self, save=True):
        for t in range(0, len(self.dataset)):
            model = self.run_one_time_period(t)
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

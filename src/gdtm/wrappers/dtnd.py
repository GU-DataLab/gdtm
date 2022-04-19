#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2014 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
# Edited to run dynamic TND and save noise distribution by Rob Churchill (Feb 2021)

r"""Python wrapper for `Dynamic Topic Noise Discriminator (D-TND)`_
adapted from `MALLET, the Java topic modelling toolkit <http://mallet.cs.umass.edu/>`_
D-TND source code can be found here: <https://github.com/GU-DataLab/topic-noise-models-source>

This module allows both D-TND model estimation from a training corpus and inference of topic distribution on new,
unseen documents, using an (optimized version of) collapsed gibbs sampling from MALLET.

"""


import logging
from gensim.utils import check_output

from .tnd import TNDMallet, eTNDMallet

logger = logging.getLogger(__name__)


class dTNDMallet(TNDMallet):
    """Python wrapper for dTND using `MALLET <http://mallet.cs.umass.edu/>`_.

    Communication between MALLET and Python takes place by passing around data files on disk
    and calling Java with subprocess.call().

    Warnings
    --------
    This is **only** python wrapper for `MALLET LDA <http://mallet.cs.umass.edu/>`_,
    you need to install original implementation first and pass the path to binary to ``mallet_path``.

        Parameters
        ----------
        mallet_path : str
            Path to the mallet binary, e.g. `/home/username/mallet-dtnd/bin/mallet`.
        corpus : iterable of iterable of (int, int), optional
            Collection of texts in BoW format.
        num_topics : int, optional
            Number of topics.
        alpha : int, optional
            Alpha parameter of LDA.
        id2word : :class:`~gensim.corpora.dictionary.Dictionary`, optional
            Mapping between tokens ids and words from corpus, if not specified - will be inferred from `corpus`.
        workers : int, optional
            Number of threads that will be used for training.
        prefix : str, optional
            Prefix for produced temporary files.
        optimize_interval : int, optional
            Optimize hyperparameters every `optimize_interval` iterations
            (sometimes leads to Java exception 0 to switch off hyperparameter optimization).
        iterations : int, optional
            Number of training iterations.
        topic_threshold : float, optional
            Threshold of the probability above which we consider a topic.
        random_seed: int, optional
            Random seed to ensure consistent results, if 0 - use system clock.
        noise_words_max: int, optional
            Number of noise words to save when saving the distribution to a file.
            The top `noise_words_max` most probable noise words will be saved.
        skew: int, optional
            Akin to the gamma parameter in the paper, increasing `skew` increases the probability of each word being
            assigned to a topic over noise
        alpha_array_infile: str, optional
            path to file containing alpha distribution from previous time period
        tw_dist_file: str, optional
            path to file containing topic-word distribution from previous time period
        noise_dist_file: str, optional
            path to file containing noise distribution from previous time period

    """
    def __init__(self, mallet_path, corpus=None, num_topics=100, alpha=50, beta=0.01, id2word=None, workers=4,
                 prefix=None, optimize_interval=0, iterations=1000, topic_threshold=0.0, random_seed=0,
                 noise_words_max=100, skew=0, alpha_array_infile=None, tw_dist_file=None, noise_dist_file=None):
        super().__init__(mallet_path, corpus=corpus, num_topics=num_topics, alpha=alpha, beta=beta, id2word=id2word,
                         workers=workers, prefix=prefix, optimize_interval=optimize_interval, iterations=iterations,
                         topic_threshold=topic_threshold, random_seed=random_seed, noise_words_max=noise_words_max,
                         skew=skew, is_parent=True)
        self.alpha_array_infile = alpha_array_infile
        self.tw_dist_file = tw_dist_file
        self.noise_dist_file = noise_dist_file
        if corpus is not None:
            self.train(corpus)

    def ftwdistfile(self):
        '''
        Get path to noise file

        :return:
        ---------
        str
        |   Path to noise distribution file.
        '''
        return self.prefix + 'tw_dist.csv'

    def falphaarrayfile(self):
        '''
        Get path to noise file

        :return:
        ---------
        str
        |   Path to noise distribution file.
        '''
        return self.prefix + 'alpha_array.csv'

    def fbetafile(self):
        '''
        Get path to noise file

        :return:
        ---------
        str
        |   Path to noise distribution file.
        '''
        return self.prefix + 'beta.csv'

    def train(self, corpus):
        """Train Mallet LDA.

        Parameters
        ----------
        corpus : iterable of iterable of (int, int)
            Corpus in BoW format

        """
        self.convert_input(corpus, infer=False)
        cmd = self.mallet_path + ' train-topics --input %s --num-topics %s --optimize-interval %s '\
            '--num-threads %s --output-state %s --output-doc-topics %s --output-topic-keys %s '\
            '--num-iterations %s --inferencer-filename %s --doc-topics-threshold %s  --random-seed %s --output-noise %s '\
            '--noise-words-max %s --skew %s --beta %s --topic-word-weights-file %s --output-alpha-array %s --output-beta %s '

        cmd = cmd % (
            self.fcorpusmallet(), self.num_topics, self.optimize_interval,
            self.workers, self.fstate(), self.fdoctopics(), self.ftopickeys(), self.iterations,
            self.finferencer(), self.topic_threshold, str(self.random_seed), self.fnoisefile(), str(self.noise_words_max),
            str(self.skew), str(self.beta), self.fwordweights(), self.falphaarrayfile(), self.fbetafile()
        )
        if self.alpha_array_infile is not None:
            cmd += '--alpha-array-infile %s ' % self.alpha_array_infile
        else:
            cmd += '--alpha %s ' % str(self.alpha)
        if self.tw_dist_file is not None:
            cmd += '--tw-dist-infile %s ' % self.tw_dist_file
        if self.noise_dist_file is not None:
            cmd += '--noise-dist-infile %s ' % self.noise_dist_file
        # NOTE "--keep-sequence-bigrams" / "--use-ngrams true" poorer results + runs out of memory
        logger.info("training MALLET LDA with %s", cmd)
        check_output(args=cmd, shell=True)
        self.word_topics = self.load_word_topics()
        # NOTE - we are still keeping the wordtopics variable to not break backward compatibility.
        # word_topics has replaced wordtopics throughout the code;
        # wordtopics just stores the values of word_topics when train is called.
        self.wordtopics = self.word_topics

    def load_alpha_array(self):
        alpha = []
        logger.info("loading alpha array from %s", self.falphaarrayfile())
        with open(self.falphaarrayfile(), 'r') as f:
            for line in f:
                prob = line.strip()
                alpha.append(float(prob))
        return alpha

    def load_beta(self):
        beta = 0
        logger.info("loading beta from %s", self.fbetafile())
        with open(self.fbetafile(), 'r') as f:
            for line in f:
                prob = line.strip()
                beta = float(prob)
        return beta

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2014 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
# Edited to run dynamic embedded TND and save noise distribution by Rob Churchill (Feb 2021)

r"""Python wrapper for `Latent Dirichlet Allocation (LDA) <https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation>`_
from `MALLET, the Java topic modelling toolkit <http://mallet.cs.umass.edu/>`_

This module allows both LDA model estimation from a training corpus and inference of topic distribution on new,
unseen documents, using an (optimized version of) collapsed gibbs sampling from MALLET.

Notes
-----
MALLET's LDA training requires :math:`O(corpus\_words)` of memory, keeping the entire corpus in RAM.
If you find yourself running out of memory, either decrease the `workers` constructor parameter,
or use :class:`gensim.models.ldamodel.LdaModel` or :class:`gensim.models.ldamulticore.LdaMulticore`
which needs only :math:`O(1)` memory.
The wrapped model can NOT be updated with new documents for online training 

"""

class deTNDMallet(eTNDMallet):
    """Python wrapper for NFT using `MALLET <http://mallet.cs.umass.edu/>`_.

    Communication between MALLET and Python takes place by passing around data files on disk
    and calling Java with subprocess.call().

    Warnings
    --------
    This is **only** python wrapper for `MALLET LDA <http://mallet.cs.umass.edu/>`_,
    you need to install original implementation first and pass the path to binary to ``mallet_path``.

    """
    def __init__(self, mallet_path, corpus=None, num_topics=100, alpha=50, beta=0.01, id2word=None, workers=4,
                 prefix=None, optimize_interval=0, iterations=1000, topic_threshold=0.0, random_seed=0,
                 noise_words_max=100, skew=0, tau=200, embedding_path=None, closest_x_words=3,
                 alpha_array_infile=None, tw_dist_file=None, noise_dist_file=None):
        """

        Parameters
        ----------
        mallet_path : str
            Path to the mallet binary, e.g. `/home/username/mallet-2.0.7/bin/mallet`.
        corpus : iterable of iterable of (int, int), optional
            Collection of texts in BoW format.
        num_topics : int, optional
            Number of topics.
        alpha : int, optional
            Alpha parameter of LDA.
        id2word : :class:`~gensim.corpora.dictionary.Dictionary`, optional
            Mapping between tokens ids and words from corpus, if not specified - will be inferred from `corpus`.
        workers : int, optional
            Number of threads that will be used for training.
        prefix : str, optional
            Prefix for produced temporary files.
        optimize_interval : int, optional
            Optimize hyperparameters every `optimize_interval` iterations
            (sometimes leads to Java exception 0 to switch off hyperparameter optimization).
        iterations : int, optional
            Number of training iterations.
        topic_threshold : float, optional
            Threshold of the probability above which we consider a topic.
        random_seed: int, optional
            Random seed to ensure consistent results, if 0 - use system clock.
        noise_words_max: int, optional
            Number of noise words to save when saving the distribution to a file.
            The top `noise_words_max` most probable noise words will be saved.
        skew: int, optional
            Akin to the gamma parameter in the paper, increasing `skew` increases the probability of each word being
            assigned to a topic over noise
        alpha_array_infile: str, optional
            path to file containing alpha distribution from previous time period
        tw_dist_file: str, optional
            path to file containing topic-word distribution from previous time period
        noise_dist_file: str, optional
            path to file containing noise distribution from previous time period
        tau: int, optional
            burn-in period for embedding sampling. The number of iterations to wait before performing embedding sampling
            on words assigned to noise.
        """
        super().__init__(mallet_path, corpus=corpus, num_topics=num_topics, alpha=alpha, beta=beta, id2word=id2word,
                         workers=workers, prefix=prefix, optimize_interval=optimize_interval, iterations=iterations,
                         topic_threshold=topic_threshold, random_seed=random_seed, noise_words_max=noise_words_max,
                         skew=skew, embedding_path=embedding_path, closest_x_words=closest_x_words, is_parent=True)
        self.alpha_array_infile = alpha_array_infile
        self.tw_dist_file = tw_dist_file
        self.noise_dist_file = noise_dist_file
        self.tau = tau
        if corpus is not None:
            self.train(corpus)

    def ftwdistfile(self):
        '''
        Get path to noise file

        :return:
        ---------
        str
        |   Path to noise distribution file.
        '''
        return self.prefix + 'tw_dist.csv'

    def falphaarrayfile(self):
        '''
        Get path to noise file

        :return:
        ---------
        str
        |   Path to noise distribution file.
        '''
        return self.prefix + 'alpha_array.csv'

    def fbetafile(self):
        '''
        Get path to noise file

        :return:
        ---------
        str
        |   Path to noise distribution file.
        '''
        return self.prefix + 'beta.csv'

    def train(self, corpus):
        """Train Mallet LDA.

        Parameters
        ----------
        corpus : iterable of iterable of (int, int)
            Corpus in BoW format

        """
        self.get_closest_embedded_words()
        self.convert_input(corpus, infer=False)
        cmd = self.mallet_path + ' train-topics --input %s --num-topics %s  --alpha %s --optimize-interval %s '\
            '--num-threads %s --output-state %s --output-doc-topics %s --output-topic-keys %s '\
            '--num-iterations %s --inferencer-filename %s --doc-topics-threshold %s  --random-seed %s ' \
            '--output-noise %s --noise-words-max %s --skew %s --beta %s --close-words-file %s --tau %s ' \
            '--optimize-burn-in %s --topic-word-weights-file %s --output-alpha-array %s --output-beta %s '

        cmd = cmd % (
            self.fcorpusmallet(), self.num_topics, self.alpha, self.optimize_interval,
            self.workers, self.fstate(), self.fdoctopics(), self.ftopickeys(), self.iterations,
            self.finferencer(), self.topic_threshold, str(self.random_seed), self.fnoisefile(),
            str(self.noise_words_max), str(self.skew), str(self.beta), self.fclosewords(), str(self.tau), str(self.tau),
            self.fwordweights(), self.falphaarrayfile(), self.fbetafile()
        )
        if self.alpha_array_infile is not None:
            cmd += '--alpha-array-infile %s ' % self.alpha_array_infile
        else:
            cmd += '--alpha %s ' % str(self.alpha)
        if self.tw_dist_file is not None:
            cmd += '--tw-dist-infile %s ' % self.tw_dist_file
        if self.noise_dist_file is not None:
            cmd += '--noise-dist-infile %s ' % self.noise_dist_file
        # NOTE "--keep-sequence-bigrams" / "--use-ngrams true" poorer results + runs out of memory
        logger.info("training MALLET LDA with %s", cmd)
        check_output(args=cmd, shell=True)
        self.word_topics = self.load_word_topics()
        # NOTE - we are still keeping the wordtopics variable to not break backward compatibility.
        # word_topics has replaced wordtopics throughout the code;
        # wordtopics just stores the values of word_topics when train is called.
        self.wordtopics = self.word_topics

    def load_alpha_array(self):
        alpha = []
        logger.info("loading alpha array from %s", self.falphaarrayfile())
        with open(self.falphaarrayfile(), 'r') as f:
            for line in f:
                prob = line.strip()
                alpha.append(float(prob))
        return alpha

    def load_beta(self):
        beta = 0
        logger.info("loading beta from %s", self.fbetafile())
        with open(self.fbetafile(), 'r') as f:
            for line in f:
                prob = line.strip()
                beta = float(prob)
        return beta

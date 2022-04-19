#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2014 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
# Edited to run Guided Topic Model and save noise distribution by Rob Churchill (March 2021)

r"""Python wrapper for `Guided Topic-Noise Model (GTM)`
adapted from `MALLET, the Java topic modelling toolkit <http://mallet.cs.umass.edu/>`
GTM source code can be found here: <https://github.com/GU-DataLab/topic-noise-models-source>

This module allows both GTM model estimation from a training corpus and inference of topic distribution on new,
unseen documents, using an (optimized version of) collapsed gibbs sampling from MALLET.

"""


import logging
from gensim.utils import check_output
from ..helpers.exceptions import MissingSeedWeightsError
from .base_wrapper import BaseMalletWrapper

logger = logging.getLogger(__name__)


class GTMMallet(BaseMalletWrapper):
    """Python wrapper for GTM using `MALLET <http://mallet.cs.umass.edu/>`.

    Communication between MALLET and Python takes place by passing around data files on disk
    and calling Java with subprocess.call().

    Warnings
    --------
    This is **only** python wrapper for `MALLET LDA <http://mallet.cs.umass.edu/>`,
    you need to install original implementation first and pass the path to binary to ``mallet_path``.

        Parameters
        ----------
        mallet_path : str
            Path to the mallet binary, e.g. `/home/username/mallet-gtm/bin/mallet`.
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
        sampling_scheme: int, optional
            0: normal sampling
            1: oversampling of observed seed word (Deprecated)
            2: GPU sampling of seed words
        over_sampling_factor: double, optional
            the amount to oversample a seed word by in sampling scheme 1.
            A value of 10 will oversample a seed word 10:1
        seed_gpu_weights: list of lists corresponding to one weight per seed word per topic

    """
    def __init__(self, mallet_path, corpus=None, num_topics=100, alpha=50, beta=0.01, id2word=None, workers=4,
                 prefix=None, optimize_interval=0, iterations=1000, topic_threshold=0.0, random_seed=0,
                 seed_topics_file=None, sampling_scheme=2, over_sampling_factor=1, seed_gpu_weights=None):
        super().__init__(mallet_path, corpus=corpus, num_topics=num_topics, alpha=alpha, beta=beta, id2word=id2word,
                         workers=workers, prefix=prefix, optimize_interval=optimize_interval, iterations=iterations,
                         topic_threshold=topic_threshold, random_seed=random_seed)
        self.seed_topics_file = seed_topics_file
        self.sampling_scheme = sampling_scheme
        self.over_sampling_factor = over_sampling_factor
        self.seed_gpu_weights = seed_gpu_weights
        if self.seed_gpu_weights is None:
            raise MissingSeedWeightsError
        self.save_seed_gpu_weights()
        if corpus is not None:
            self.train(corpus)

    def fnoisefile(self):
        '''
        Get path to noise file

        :return:
        ---------
        str
        |   Path to noise distribution file.
        '''
        return self.prefix + 'noise_dist.csv'

    def fseedgpuweights(self):
        """Get path to word weight file.

        Returns
        -------
        str
            Path to word weight file.

        """
        return self.prefix + 'seedgpuweights.txt'

    def save_seed_gpu_weights(self):
        with open(self.fseedgpuweights(), 'w') as f:
            if self.seed_gpu_weights is not None:
                for weights in self.seed_gpu_weights:
                    w_strs = ["%.2f" % x for x in weights]
                    f.write('{}\n'.format(','.join(w_strs)))

    def train(self, corpus):
        """Train Mallet LDA.

        Parameters
        ----------
        corpus : iterable of iterable of (int, int)
            Corpus in BoW format

        """
        self.convert_input(corpus, infer=False)
        cmd = self.mallet_path + ' train-topics --input %s --num-topics %s  --alpha %s --optimize-interval %s '\
            '--num-threads %s --output-state %s --output-doc-topics %s --output-topic-keys %s '\
            '--num-iterations %s --inferencer-filename %s --doc-topics-threshold %s  --random-seed %s --beta %s ' \
            '--samplingScheme %s --overSamplingFactor %s --seed-gpu-weights %s'

        cmd = cmd % (
            self.fcorpusmallet(), self.num_topics, self.alpha, self.optimize_interval,
            self.workers, self.fstate(), self.fdoctopics(), self.ftopickeys(), self.iterations,
            self.finferencer(), self.topic_threshold, str(self.random_seed), str(self.beta),
            str(self.sampling_scheme), str(self.over_sampling_factor), self.fseedgpuweights()
        )
        if self.seed_topics_file is not None:
            cmd += '  --seed-topics %s' % self.seed_topics_file
        # NOTE "--keep-sequence-bigrams" / "--use-ngrams true" poorer results + runs out of memory
        logger.info("training MALLET LDA with %s", cmd)
        check_output(args=cmd, shell=True)
        self.word_topics = self.load_word_topics()
        # NOTE - we are still keeping the wordtopics variable to not break backward compatibility.
        # word_topics has replaced wordtopics throughout the code;
        # wordtopics just stores the values of word_topics when train is called.
        self.wordtopics = self.word_topics

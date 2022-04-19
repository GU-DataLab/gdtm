#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2014 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


r"""Python wrapper for `Dynamic Latent Dirichlet Allocation (dLDA) <https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation>`_
from `MALLET, the Java topic modelling toolkit <http://mallet.cs.umass.edu/>`_

This module allows both LDA model estimation from a training corpus and inference of topic distribution on new,
unseen documents, using an (optimized version of) collapsed gibbs sampling from MALLET.

Notes
-----
MALLET's LDA training requires :math:`O(corpus\_words)` of memory, keeping the entire corpus in RAM.
If you find yourself running out of memory, either decrease the `workers` constructor parameter,
or use :class:`gensim.models.ldamodel.LdaModel` or :class:`gensim.models.ldamulticore.LdaMulticore`
which needs only :math:`O(1)` memory.
The wrapped model can NOT be updated with new documents for online training -- use
:class:`~gensim.models.ldamodel.LdaModel` or :class:`~gensim.models.ldamulticore.LdaMulticore` for that.

Installation
------------
Use `official guide <http://mallet.cs.umass.edu/download.php>`_ or this one ::

    sudo apt-get install default-jdk
    sudo apt-get install ant
    git clone git@github.com:mimno/Mallet.git
    cd Mallet/
    ant

Examples
--------
.. sourcecode:: pycon

    >>> from gensim.test.utils import common_corpus, common_dictionary
    >>> from gensim.models.wrappers import LdaMallet
    >>>
    >>> path_to_mallet_binary = "/path/to/mallet/binary"
    >>> model = LdaMallet(path_to_mallet_binary, corpus=common_corpus, num_topics=20, id2word=common_dictionary)
    >>> vector = model[common_corpus[0]]  # LDA topics of a documents

"""


import logging
from gensim.utils import check_output
from .base_wrapper import BaseMalletWrapper

logger = logging.getLogger(__name__)


class dLDAMallet(BaseMalletWrapper):
    """Python wrapper for dynamic LDA using `MALLET <http://mallet.cs.umass.edu/>`_.
    D-LDA source code can be found here: <https://github.com/GU-DataLab/topic-noise-models-source>

    Communication between MALLET and Python takes place by passing around data files on disk
    and calling Java with subprocess.call().

    Warnings
    --------
    This is **only** python wrapper for `MALLET LDA <http://mallet.cs.umass.edu/>`_,
    you need to install original implementation first and pass the path to binary to ``mallet_path``.

        Parameters
        ----------
        mallet_path : str
            Path to the mallet binary, e.g. `/home/username/mallet-dlda/bin/mallet`.
        corpus : iterable of iterable of (int, int), optional
            Collection of texts in BoW format.
        num_topics : int, optional
            Number of topics.
        alpha : int, optional
            Initial Alpha parameter of d-LDA.
        beta: float, optional
            Initial Beta parameter of d-LDA.
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
        alpha_array_infile: str, optional
            path to file containing alpha distribution from previous time period
        tw_dist_file: str, optional
            path to file containing topic-word distribution from previous time period

    """
    def __init__(self, mallet_path, corpus=None, num_topics=100, alpha=50, beta=0.01, id2word=None, workers=4,
                 prefix=None, optimize_interval=0, iterations=1000, topic_threshold=0.0, random_seed=0,
                 alpha_array_infile=None, tw_dist_file=None):
        super().__init__(mallet_path, corpus=corpus, num_topics=num_topics, alpha=alpha, beta=beta, id2word=id2word,
                         workers=workers, prefix=prefix, optimize_interval=optimize_interval, iterations=iterations,
                         topic_threshold=topic_threshold, random_seed=random_seed)
        self.alpha_array_infile = alpha_array_infile
        self.tw_dist_file = tw_dist_file
        if corpus is not None:
            self.train(corpus)


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
        cmd = self.mallet_path + ' train-topics --input %s --num-topics %s  --alpha %s --optimize-interval %s '\
            '--num-threads %s --output-state %s --output-doc-topics %s --output-topic-keys %s '\
            '--num-iterations %s --inferencer-filename %s --doc-topics-threshold %s  --random-seed %s '\
            '--topic-word-weights-file %s --output-alpha-array %s --beta %s --output-beta %s '

        cmd = cmd % (
            self.fcorpusmallet(), self.num_topics, self.alpha, self.optimize_interval,
            self.workers, self.fstate(), self.fdoctopics(), self.ftopickeys(), self.iterations,
            self.finferencer(), self.topic_threshold, str(self.random_seed), self.fwordweights(),
            self.falphaarrayfile(), str(self.beta), self.fbetafile()
        )
        if self.alpha_array_infile is not None:
            cmd += '--alpha-array-infile %s ' % self.alpha_array_infile
        else:
            cmd += '--alpha %s ' % str(self.alpha)
        if self.tw_dist_file is not None:
            cmd += '--tw-dist-infile %s ' % self.tw_dist_file
        # NOTE "--keep-sequence-bigrams" / "--use-ngrams true" poorer results + runs out of memory
        logger.info("training MALLET LDA with %s", cmd)
        check_output(args=cmd, shell=True)
        self.word_topics = self.load_word_topics()

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

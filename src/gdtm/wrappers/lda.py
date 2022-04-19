#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2014 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
# Refactored by Rob Churchill <rjc111@georgetown.edu>

r"""Python wrapper for `Latent Dirichlet Allocation (LDA) <https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation>`
from `MALLET, the Java topic modelling toolkit <http://mallet.cs.umass.edu/>`

This module allows both LDA model estimation from a training corpus and inference of topic distribution on new,
unseen documents, using an (optimized version of) collapsed gibbs sampling from MALLET.

Notes
-----
MALLET's LDA training requires :math:`O(corpus\_words)` of memory, keeping the entire corpus in RAM.
The wrapped model can NOT be updated with new documents for online training

Installation
------------
Use `official guide <http://mallet.cs.umass.edu/download.php>` or this one ::

    sudo apt-get install default-jdk
    sudo apt-get install ant
    git clone git@github.com:mimno/Mallet.git
    cd Mallet/
    ant

"""


import logging
from gensim.utils import check_output
from .base_wrapper import BaseMalletWrapper

logger = logging.getLogger(__name__)


class LDAMallet(BaseMalletWrapper):
    """Python wrapper for LDA using `MALLET <http://mallet.cs.umass.edu/>`.

    Communication between MALLET and Python takes place by passing around data files on disk
    and calling Java with subprocess.call().

    Warnings
    --------
    This is **only** python wrapper for `MALLET LDA <http://mallet.cs.umass.edu/>`,
    you need to install original implementation first and pass the path to binary to ``mallet_path``.

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

    """
    def __init__(self, mallet_path, corpus=None, num_topics=100, alpha=50, id2word=None, workers=4,
                 prefix=None, optimize_interval=0, iterations=1000, topic_threshold=0.0, random_seed=0):
        super().__init__(mallet_path, corpus=corpus, num_topics=num_topics, alpha=alpha, id2word=id2word,
                         workers=workers, prefix=prefix, optimize_interval=optimize_interval, iterations=iterations,
                         topic_threshold=topic_threshold, random_seed=random_seed)
        if corpus is not None:
            self.train(corpus)

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
            '--num-iterations %s --inferencer-filename %s --doc-topics-threshold %s  --random-seed %s'

        cmd = cmd % (
            self.fcorpusmallet(), self.num_topics, self.alpha, self.optimize_interval,
            self.workers, self.fstate(), self.fdoctopics(), self.ftopickeys(), self.iterations,
            self.finferencer(), self.topic_threshold, str(self.random_seed)
        )
        # NOTE "--keep-sequence-bigrams" / "--use-ngrams true" poorer results + runs out of memory
        logger.info("training MALLET LDA with %s", cmd)
        check_output(args=cmd, shell=True)
        self.word_topics = self.load_word_topics()
        # NOTE - we are still keeping the wordtopics variable to not break backward compatibility.
        # word_topics has replaced wordtopics throughout the code;
        # wordtopics just stores the values of word_topics when train is called.
        self.wordtopics = self.word_topics

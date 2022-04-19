#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2014 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
# Refactored to run TND and save noise distribution by Rob Churchill (Feb 2021)

r"""Python wrapper for `Topic Noise Discriminator (TND)`
adapted from `MALLET, the Java topic modelling toolkit <http://mallet.cs.umass.edu/>`
TND source code can be found here: <https://github.com/GU-DataLab/topic-noise-models-source>

This module allows both TND model estimation from a training corpus and inference of topic distribution on new,
unseen documents, using an (optimized version of) collapsed gibbs sampling from MALLET.

"""


import logging
import os
from gensim.utils import check_output
from gensim.models.fasttext import load_facebook_vectors
from .base_wrapper import BaseMalletWrapper

logger = logging.getLogger(__name__)


class TNDMallet(BaseMalletWrapper):
    """Python wrapper for TND using `MALLET <http://mallet.cs.umass.edu/>`.
    TND's source code can be found here: <https://github.com/GU-DataLab/topic-noise-models-source>.

    Communication between MALLET and Python takes place by passing around data files on disk
    and calling Java with subprocess.call().

    Warnings
    --------
    This is **only** python wrapper for `MALLET TND`,
    you need to install original implementation first and pass the path to binary to ``mallet_path``.

        Parameters
        ----------
        mallet_path : str
            Path to the mallet binary, e.g. `/home/username/mallet-tnd/bin/mallet`.
        corpus : iterable of iterable of (int, int), optional
            Collection of texts in BoW format.
        num_topics : int, optional
            Number of topics.
        alpha : int, optional
            Alpha parameter of TND.
        beta : float, optional
            Beta parameter of TND
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

    """
    def __init__(self, mallet_path, corpus=None, num_topics=100, alpha=50, beta=0.01, id2word=None, workers=4,
                 prefix=None, optimize_interval=0, iterations=1000, topic_threshold=0.0, random_seed=0,
                 noise_words_max=100, skew=0, is_parent=False):
        super().__init__(mallet_path, corpus=corpus, num_topics=num_topics, alpha=alpha, beta=beta, id2word=id2word,
                         workers=workers, prefix=prefix, optimize_interval=optimize_interval, iterations=iterations,
                         topic_threshold=topic_threshold, random_seed=random_seed)
        self.noise_words_max = noise_words_max
        self.skew = skew
        if corpus is not None and not is_parent:
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
            '--num-iterations %s --inferencer-filename %s --doc-topics-threshold %s  --random-seed %s --output-noise %s '\
            '--noise-words-max %s --skew %s --beta %s'

        cmd = cmd % (
            self.fcorpusmallet(), self.num_topics, self.alpha, self.optimize_interval,
            self.workers, self.fstate(), self.fdoctopics(), self.ftopickeys(), self.iterations,
            self.finferencer(), self.topic_threshold, str(self.random_seed), self.fnoisefile(), str(self.noise_words_max),
            str(self.skew), str(self.beta)
        )
        # NOTE "--keep-sequence-bigrams" / "--use-ngrams true" poorer results + runs out of memory
        logger.info("training MALLET LDA with %s", cmd)
        check_output(args=cmd, shell=True)
        self.word_topics = self.load_word_topics()
        # NOTE - we are still keeping the wordtopics variable to not break backward compatibility.
        # word_topics has replaced wordtopics throughout the code;
        # wordtopics just stores the values of word_topics when train is called.
        self.wordtopics = self.word_topics

    def load_noise_dist(self):
        noise_dist = {}
        logger.info("loading assigned topics from %s", self.fnoisefile())
        with open(self.fnoisefile(), 'r') as f:
            for line in f:
                word, val = line.strip().split(', ')
                noise_dist[word] = int(val)
        return noise_dist


r"""Python wrapper for `embedded TND `
from `MALLET, the Java topic modelling toolkit <http://mallet.cs.umass.edu/>`

This module allows both eTND model estimation from a training corpus and inference of topic distribution on new,
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
Use `official guide <http://mallet.cs.umass.edu/download.php>` or this one ::

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


class eTNDMallet(TNDMallet):
    """Python wrapper for eTND using `MALLET <http://mallet.cs.umass.edu/>`.

    Communication between MALLET and Python takes place by passing around data files on disk
    and calling Java with subprocess.call().

    Warnings
    --------
    This is **only** python wrapper for `MALLET LDA <http://mallet.cs.umass.edu/>`,
    you need to install original implementation first and pass the path to binary to ``mallet_path``.

    """
    def __init__(self, mallet_path, corpus=None, num_topics=100, alpha=50, beta=0.01, id2word=None, workers=4, prefix=None,
                 optimize_interval=0, iterations=1000, topic_threshold=0.0, random_seed=0, noise_words_max=100, skew=0,
                 tau=200, embedding_path=None, closest_x_words=3, is_parent=False):
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
        tau: int, optional
            burn-in period for embedding sampling. The number of iterations to wait before performing embedding sampling
            on words assigned to noise.
        embedding_path: str, optional
            Path to word embeddings file.  Required if using word embedding sampling.
        closest_x_words: int, optional
            The number of words to sample from the embedding space for an assigned noise word. Increasing this value
            leads to more words being added to the noise distribution.
        """
        super().__init__(mallet_path, corpus=corpus, num_topics=num_topics, alpha=alpha, beta=beta, id2word=id2word,
                         workers=workers, prefix=prefix, optimize_interval=optimize_interval, iterations=iterations,
                         topic_threshold=topic_threshold, random_seed=random_seed, noise_words_max=noise_words_max,
                         skew=skew, is_parent=True)
        self.tau = tau
        self.embedding_path = embedding_path
        self.closest_x_words = closest_x_words
        if corpus is not None and not is_parent:
            self.train(corpus)

    def fclosewords(self):
        """Get path to closest words to each word in embedding space

        Returns
        -------
        str
        """
        # if os.path.exists('data/cw_{}_{}.txt'.format(self.embedding_path.replace('.bin', ''), self.closest_x_words)):
        #     return 'data/cw_{}'.format(self.embedding_path.replace('.bin', '.txt'))
        return self.prefix + 'closewords.txt'

    def get_closest_embedded_words(self):
        if os.path.exists('data/cw_{}_{}.txt'.format(self.embedding_path.replace('.bin', ''), self.closest_x_words)):
            return
        vec = load_facebook_vectors(self.embedding_path)
        vec.fill_norms()
        hasht = dict()
        for w in vec.key_to_index:
            hasht[w] = [x[0] for x in vec.most_similar_cosmul(w, topn=self.closest_x_words)]
        with open(self.fclosewords(), 'w') as f:
            for w in hasht.keys():
                f.write('{},{}\n'.format(w, ','.join(hasht[w])))

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
            '--num-iterations %s --inferencer-filename %s --doc-topics-threshold %s  --random-seed %s --output-noise %s '\
            '--noise-words-max %s --skew %s --beta %s --close-words-file %s --tau %s --optimize-burn-in %s'

        cmd = cmd % (
            self.fcorpusmallet(), self.num_topics, self.alpha, self.optimize_interval,
            self.workers, self.fstate(), self.fdoctopics(), self.ftopickeys(), self.iterations,
            self.finferencer(), self.topic_threshold, str(self.random_seed), self.fnoisefile(), str(self.noise_words_max),
            str(self.skew), str(self.beta), self.fclosewords(), str(self.tau), str(self.tau)
        )
        # NOTE "--keep-sequence-bigrams" / "--use-ngrams true" poorer results + runs out of memory
        logger.info("training MALLET LDA with %s", cmd)
        check_output(args=cmd, shell=True)
        self.word_topics = self.load_word_topics()
        # NOTE - we are still keeping the wordtopics variable to not break backward compatibility.
        # word_topics has replaced wordtopics throughout the code;
        # wordtopics just stores the values of word_topics when train is called.
        self.wordtopics = self.word_topics

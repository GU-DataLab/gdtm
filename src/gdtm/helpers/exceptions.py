'''
Exceptions shared across files.
'''

class MissingModelError(Exception):
    '''
    Error for model classes that are initialized without the underlying Java model or a pretrained model.

    '''
    def __init__(self, model, message=None):
        self.model = model
        self.message = message
        if self.message is None:
            self.message = 'You must either define the path to the {} model, or provide the pretrained model.'
        super().__init__(self.message.format(model))


class MissingDataSetError(Exception):
    '''
    Error for model classes that are initialized without a data set or a pretrained model.

    '''
    def __init__(self, message=None):
        self.message = message
        if self.message is None:
            self.message = 'You must either provide a pretrained noise distribution and pretrained topic-word ' \
                           'distribution, or provide a data set to be modeled.'
        super().__init__(self.message)


class MissingSeedsError(Exception):
    '''
    Error for Guided Topic-Noise model class that is initialized without seed topics.

    '''
    def __init__(self, message=None):
        self.message = message
        if self.message is None:
            self.message = 'You must provide a seed topics file.'
        super().__init__(self.message)


class MissingSeedWeightsError(Exception):
    '''
    Error for GTM wrapper class that is initialized without seed weights.

    '''
    def __init__(self, message=None):
        self.message = message
        if self.message is None:
            self.message = 'You must provide a seed word weighting map.  ' \
                           'You can do this easily using: helpers.weighting.compute_idf_weights'
        super().__init__(self.message)


class MissingEmbeddingPathError(Exception):
    '''
    Error for embedded model classes that are initialized without a path to embedding vectors.

    '''
    def __init__(self, message=None):
        self.message = message
        if self.message is None:
            self.message = 'You must provide a path to a pre-trained embedding space.'
        super().__init__(self.message)


class NotEnoughTopicSetsError(Exception):
    '''
    Error for CSTB functions that are not provided at least two topic sets for blending.

    '''
    def __init__(self, message=None):
        self.message = message
        if self.message is None:
            self.message = 'You must input at least two topic sets to perform CSTB.'
        super().__init__(self.message)

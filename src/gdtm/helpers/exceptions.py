'''
Exceptions shared across files.
'''

class MissingModelError(Exception):
    def __init__(self, model, message=None):
        self.model = model
        self.message = message
        if self.message is None:
            self.message = 'You must either define the path to the {} model, or provide the pretrained model.'
        super().__init__(self.message.format(model))


class MissingDataSetError(Exception):
    def __init__(self, message=None):
        self.message = message
        if self.message is None:
            self.message = 'You must either provide a pretrained noise distribution and pretrained topic-word ' \
                           'distribution, or provide a data set to be modeled.'
        super().__init__(self.message)


class MissingSeedsError(Exception):
    def __init__(self, message=None):
        self.message = message
        if self.message is None:
            self.message = 'You must provide a seed topics file.'
        super().__init__(self.message)


class MissingSeedWeightsError(Exception):
    def __init__(self, message=None):
        self.message = message
        if self.message is None:
            self.message = 'You must provide a seed word weighting map.  ' \
                           'You can do this easily using: helpers.weighting.compute_idf_weights'
        super().__init__(self.message)


class MissingEmbeddingPathError(Exception):
    def __init__(self, message=None):
        self.message = message
        if self.message is None:
            self.message = 'You must provide a path to a pre-trained embedding space.'
        super().__init__(self.message)


class NotEnoughTopicSetsError(Exception):
    def __init__(self, message=None):
        self.message = message
        if self.message is None:
            self.message = 'You must input at least two topic sets to perform CSTI.'
        super().__init__(self.message)

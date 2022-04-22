

def compute_idf_weights(dataset, topics):
    '''
    For each topic word, computes the inverse document frequency (IDF) of the words for seed topic weighting.

    :param dataset: list of lists
    :param topics: list of lists (topic words)
    :return: list of lists (weights of topic words)
    '''
    weights = []
    num = len(dataset)
    for topic in topics:
        weight_list = []
        for w in topic:
            df = 1 + sum([1 for d in dataset if w in d])
            weight_list.append(num/df)
        weights.append(weight_list)
    return weights

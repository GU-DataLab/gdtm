

def compute_idf_weights(dataset, topics):
    weights = []
    num = len(dataset)
    for topic in topics:
        weight_list = []
        for w in topic:
            df = 1 + sum([1 for d in dataset if w in d])
            weight_list.append(num/df)
        weights.append(weight_list)
    return weights

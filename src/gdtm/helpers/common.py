from datetime import datetime, timedelta


def load_flat_dataset(path, delimiter=' '):
    dataset = []
    with open(path, 'r') as f:
        for line in f:
            dataset.append(line.strip().split(delimiter))
    return dataset


def save_flat_dataset(path, dataset, delimiter=' '):
    with open(path, 'w') as f:
        for d in dataset:
            f.write('{}\n'.format(delimiter.join(d)))


def load_dated_dataset(path, date_delimiter='\t', doc_delimiter=' '):
    dataset = []
    with open(path, 'r') as f:
        for line in f:
            data = line.strip().split(date_delimiter)
            date, doc = data[0], data[1]
            doc = doc.split(doc_delimiter)
            dataset.append((date, doc))
    return dataset


def save_split_dataset(path, file_name, dataset, delimiter=' '):
    '''
    Saves a data set sliced into tranches for temporal modeling.

    :param path: PATH SHOULD BE A PATH AND NOT A FILE NAME
    :param file_name: the name of the data set
    :param dataset:
    :param delimiter:
    :return: None
    '''
    for i in range(0, len(dataset)):
        with open('{}{}_{}.csv'.format(path, file_name, i), 'w') as f:
            for d in dataset[i]:
                f.write('{}\n'.format(delimiter.join(d)))


def load_split_dataset(path, file_name, num_time_periods, delimiter=' '):
    dataset = []
    for i in range(0, num_time_periods):
        docs = []
        with open('{}{}_{}.csv'.format(path, file_name, i), 'r') as f:
            for line in f:
                docs.append(line.strip().split(delimiter))
        dataset.append(docs)
    return dataset


def year(d):
    return datetime(year=d.year, month=1, day=1)


def month(d):
    return datetime(year=d.year, month=d.month, day=1)


def week(d):
    diff = d.weekday()
    return d if diff == 6 else d - timedelta(days=diff + 1)


def fortnight(d):
    return d.isocalendar()[1] - (1 + d.isocalendar()[1] % 2)


def day(d):
    return d.date()


def none(d):
    return 1


def get_time_periods(dataset, epoch_function):
    times = list(set([epoch_function(x[0]) for x in dataset]))
    return sorted(times)


def split_dataset_by_date(dataset, epoch_function):
    split_dataset = []
    time_periods = get_time_periods(dataset, epoch_function)
    [split_dataset.append([]) for x in time_periods]
    for d in dataset:
        split_dataset[time_periods.index(epoch_function[d[0]])].append(d[1])
    return split_dataset


def get_vocabulary(docs):
    '''
    This version of get_vocabulary takes 0.08 seconds on 100,000 documents whereas the old version took forever.
    '''
    vocab = []
    for i in range(0, len(docs)):
        vocab.extend(docs[i])
    return list(set(vocab))


def save_topics(topics, path, delimiter=','):
    with open(path, 'w') as f:
        for topic in topics:
            f.write('{}\n'.format(delimiter.join(topic)))


def save_noise_dist(noise, path):
    with open(path, 'w') as f:
        for word, freq in noise:
            f.write('{},{}\n'.format(word, freq))


def load_topics(path, delimiter=','):
    topics = []
    with open(path, 'r') as f:
        for line in f:
            words = line.strip().split(delimiter)
            words = [w for w in words if len(w) > 0]
            topics.append(words)
    return topics


def load_noise_dist(path, delimiter=','):
    noise_dist = []
    with open(path, 'r') as f:
        for line in f:
            word, freq = line.strip().split(delimiter)
            noise_dist.append((word, freq))
    return noise_dist

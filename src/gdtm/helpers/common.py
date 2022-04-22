from datetime import datetime, timedelta


def load_flat_dataset(path, delimiter=' '):
    '''
    Load a data set from a CSV.
        Each document should be its own line, with each word in a document separated by the delimiter character.

    :param path: path (including file name) of data set file
    :param delimiter: character, optional:
        character delimiting words in a document
    :return: list of lists
    '''
    dataset = []
    with open(path, 'r') as f:
        for line in f:
            dataset.append(line.strip().split(delimiter))
    return dataset


def save_flat_dataset(path, dataset, delimiter=' '):
    '''
    Save a data set to a CSV.
        Each data set is saved to its own line in the CSV, with each word separated by the delimiter character.

    :param path: path (including file name) of data set file
    :param dataset: list of lists
    :param delimiter: character, optional:
        character delimiting words in a document
    :return:
    '''
    with open(path, 'w') as f:
        for d in dataset:
            f.write('{}\n'.format(delimiter.join(d)))


def load_dated_dataset(path, date_delimiter='\t', doc_delimiter=' '):
    '''
    Load a data set with dates from a CSV.
        The file format should be one line per document, with the date field, followed by the document. The date and
        document delimiters should be different.
        For example: 09/10/2021[tab]word1,word2,word3...wordn

    :param path: path (including file name) of data set file
    :param date_delimiter: character, optional:
        character delimiting date from docuemnt on a line
    :param doc_delimiter: character, optional:
        character delimiting words in a document
    :return: list of tuples (date, document [list of words])
    '''
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
    Save a data set sliced into tranches for temporal modeling.

    :param path: path (including file name) of data set file
    :param file_name: the name of the data set
    :param dataset: list of lists of lists
        (list of time periods, containing list of documents, containing list of words)
    :param delimiter: character, optional:
        character delimiting words in a document
    :return: None
    '''
    for i in range(0, len(dataset)):
        with open('{}{}_{}.csv'.format(path, file_name, i), 'w') as f:
            for d in dataset[i]:
                f.write('{}\n'.format(delimiter.join(d)))


def load_split_dataset(path, file_name, num_time_periods, delimiter=' '):
    '''
    Load a data set sliced into tranches for temporal modeling.

    :param path: path (including file name) of data set file
    :param file_name: the name of the data set
    :param num_time_periods: the number of time periods that the data set has been sliced into (tranches)
    :param delimiter: character, optional:
        character delimiting words in a document
    :return: list of lists of lists
        (list of time periods, containing list of documents, containing list of words)
    '''
    dataset = []
    for i in range(0, num_time_periods):
        docs = []
        with open('{}{}_{}.csv'.format(path, file_name, i), 'r') as f:
            for line in f:
                docs.append(line.strip().split(delimiter))
        dataset.append(docs)
    return dataset


def year(d):
    '''
    *Epoch function.*
    Takes a date and normalizes it to the nearest year.
    Every date from 2021 will return January 1, 2021.

    :param d: Datetime object
    :return: Datetime object
    '''
    return datetime(year=d.year, month=1, day=1)


def month(d):
    '''
    *Epoch function.*
    Takes a date and normalizes it to the nearest year and month.
    Every date from January 2021 will return January 1, 2021.

    :param d: Datetime object
    :return: Datetime object
    '''
    return datetime(year=d.year, month=d.month, day=1)


def week(d):
    '''
    *Epoch function.*
    Takes a date and normalizes it to the first day of its week.
    Every date from January 4-10, 2021 will return January 4, 2021.

    :param d: Datetime object
    :return: Datetime object
    '''
    diff = d.weekday()
    return d if diff == 6 else d - timedelta(days=diff + 1)


def fortnight(d):
    '''
    *Epoch function.*
    Takes a date and normalizes it to the first day of its fortnight (two-week period).
    Every date from January 4-17, 2021 will return January 4, 2021.
    Note: Unsure if this is a valid date range for a fortnight, or if we are off by a week.

    :param d: Datetime object
    :return: Datetime object
    '''
    return d.isocalendar()[1] - (1 + d.isocalendar()[1] % 2)


def day(d):
    '''
    *Epoch function.*
    Takes a date and normalizes it to the day (removing hour, minute, second).
    Every datetime object from January 1, 2021 (e.g. January 1, 2021 03:12:48, January 1, 2021 12:45:00)
    will return January 1, 2021.

    :param d: Datetime object
    :return: Datetime object
    '''
    return d.date()


def none(d):
    '''
    *Epoch function.*
    The null date normalizer.  Normalizes every date to 1.

    :param d: Datetime object
    :return: 1
    '''
    return 1


def get_time_periods(dataset, epoch_function):
    '''
    Computes all valid time periods in a data set, given the dates and an epoch function.

    :param dataset: list of lists
    :param epoch_function: a function that normalizes dates to a time period
    :return: sorted list of time periods
    '''
    times = list(set([epoch_function(x[0]) for x in dataset]))
    return sorted(times)


def split_dataset_by_date(dataset, epoch_function):
    '''
    Takes a data set and epoch function, and splits the data set into tranches based on the valid time periods.

    :param dataset: list of lists
    :param epoch_function: a function that normalizes dates to a time period
    :return: list of lists of lists (data set split into time periods)
    '''
    split_dataset = []
    time_periods = get_time_periods(dataset, epoch_function)
    [split_dataset.append([]) for x in time_periods]
    for d in dataset:
        split_dataset[time_periods.index(epoch_function[d[0]])].append(d[1])
    return split_dataset


def get_vocabulary(docs):
    '''
    Given a set of documents, computes the unique vocabulary.
    This version of get_vocabulary takes 0.08 seconds on 100,000 documents whereas the old version took forever.

    :param docs: list of lists
    :return: list of unique words
    '''
    vocab = []
    for i in range(0, len(docs)):
        vocab.extend(docs[i])
    return list(set(vocab))


def save_topics(topics, path, delimiter=','):
    '''
    Saves a topic set to a CSV.

    :param topics: list of lists
    :param path: path (including file name) to save topic set to.
    :param delimiter: character, optional:
        character delimiting words in a document
    :return:
    '''
    with open(path, 'w') as f:
        for topic in topics:
            f.write('{}\n'.format(delimiter.join(topic)))


def save_noise_dist(noise, path):
    '''
    Save the words in a noise distribution to a CSV.

    :param noise: list of tuples (word, frequency)
    :param path: path (including file name) to save noise distribution to.
    :return:
    '''
    with open(path, 'w') as f:
        for word, freq in noise:
            f.write('{},{}\n'.format(word, freq))


def load_topics(path, delimiter=','):
    '''
    Load topic set from a CSV.

    :param path: path (including file name) to save topic set to.
    :param delimiter: character, optional:
        character delimiting words in a document
    :return: list of lists
    '''
    topics = []
    with open(path, 'r') as f:
        for line in f:
            words = line.strip().split(delimiter)
            words = [w for w in words if len(w) > 0]
            topics.append(words)
    return topics


def load_noise_dist(path, delimiter=','):
    '''
    Load a noise distribution from a CSV.

    :param path: path (including file name) to save topic set to.
    :param delimiter: character, optional:
        character delimiting words in a document
    :return: list of tuples (word, frequency)
    '''
    noise_dist = []
    with open(path, 'r') as f:
        for line in f:
            word, freq = line.strip().split(delimiter)
            noise_dist.append((word, freq))
    return noise_dist

'''
Cross-Source Topic Blending (CSTB)

'''


from ..helpers.common import load_topics, save_topics
from ..helpers.exceptions import NotEnoughTopicSetsError


def _match_topics(t1, t2, word_threshold=3, topn=5):
    set_t1 = set(t1[:topn])
    set_t2 = set(t2[:topn])
    if len(set_t1 & set_t2) >= word_threshold:
        return True
    return False


def _filter_components(components, source_threshold=3, num_sources=3):
    filtered_components = []
    for comp in components:
        if len(comp) < source_threshold:
            continue
        if len(comp) > num_sources:
            continue
        topic_set_numbers = set([x[0] for x in comp])
        if len(topic_set_numbers) != len(comp):
            continue
        filtered_components.append(comp)
    return filtered_components


def _combine_topic(topics, words_per_topic=5):
    '''
        Threads lists into each other (first word from each list, then second, third...)
        Then removes duplicates
    '''
    combined_topic = []
    rounds = 0
    while rounds < words_per_topic:
        for i in range(0, len(topics)):
            topic = topics[i]
            if len(topic) > rounds:
                combined_topic.append(topic[rounds])
        rounds += 1
    return list(dict.fromkeys(combined_topic))


def _combine_topics(matched_topics, words_per_topic=5):
    topics = []
    for matched_topic in matched_topics:
        topic = _combine_topic(matched_topic, words_per_topic=words_per_topic)
        topics.append(topic)
    return topics


def cstb(topic_sets, source_thresholds=(3,), word_threshold=3, topn=5, words_per_topic=10):
    '''
    Cross-Source Topic Blending: Blends topics by finding them in other data sources.

    :param topic_sets: list of topic sets, one for each data source.
    :param source_thresholds: list, required:
        minimum number of sources that a topic must appear in to be integrated (ell in paper)
    :param word_threshold: int, required:
        minimum number of words overlapping to match a topic between sources (chi in paper)
    :param topn: int, optional:
        number of words per topic to consider for blending (psi in paper)
    :param words_per_topic: int, optional:
        number of words per topic to put into final topic
    '''

    topic_components = []
    if len(topic_sets) < 2:
        raise NotEnoughTopicSetsError()
    num_sources = len(topic_sets)

    for i in range(0, len(topic_sets)):
        ts_i = topic_sets[i]
        for m in range(0, len(ts_i)):
            t1 = ts_i[m]
            for j in range(i+1, len(topic_sets)):
                ts_j = topic_sets[j]
                for n in range(0, len(ts_j)):
                    t2 = ts_j[n]
                    if _match_topics(t1, t2, word_threshold=word_threshold, topn=topn):
                        comp_exists = False
                        for comp in topic_components:
                            if (i, m) in comp:
                                if not (j, n) in comp:
                                    comp.append((j, n))
                                comp_exists = True
                                break
                            elif (j, n) in comp:
                                if not (i, m) in comp:
                                    comp.append((i, m))
                                comp_exists = True
                                break
                        if not comp_exists:
                            topic_components.append([(i, m), (j, n)])

    core_topic_sets = []
    for source_threshold in source_thresholds:
        core_components = _filter_components(topic_components, source_threshold, num_sources)
        core_topics = []
        for comp in core_components:
            topics = []
            for topic in comp:
                topics.append(topic_sets[topic[0]][topic[1]])
            core_topics.append(topics)
        core_topics = _combine_topics(core_topics, words_per_topic=words_per_topic)
        core_topic_sets.append(core_topics)
    return core_topic_sets


def _main():
    dataset_label = 'covid'
    dataset_names = ['covid_election2020', 'covid_news', 'covid_reddit']
    results_path = 'results/lmm_paper'
    model = 'nlda'
    source_thresholds = [2, 3]
    word_threshold = 10
    topn = 30
    words_per_topic = 10

    topic_sets = []
    for dataset_name in dataset_names:
        topics = load_topics('{}/{}/{}/topics_best_params.csv'.format(results_path, dataset_name, model))
        topic_sets.append(topics)

    core_topic_sets = cstb(topic_sets, source_thresholds=source_thresholds, word_threshold=word_threshold, topn=topn,
                           words_per_topic=words_per_topic)
    for i in range(0, len(core_topic_sets)):
        core_topics = core_topic_sets[i]
        source_threshold = source_thresholds[i]
        st_str = '{}_{}_{}'.format(source_threshold, word_threshold, topn)
        save_topics(core_topics, '{}/{}_{}_CSTI.csv'.format(results_path, dataset_label, st_str))


if __name__ == '__main__':
    _main()

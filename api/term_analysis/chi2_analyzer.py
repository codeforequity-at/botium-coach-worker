import re
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from api.utils import term_data
import multiprocessing as mp
from multiprocessing.reduction import ForkingPickler, AbstractReducer
from concurrent.futures import ThreadPoolExecutor

class ForkingPickler4(ForkingPickler):
    def __init__(self, *args):
        if len(args) > 1:
            args[1] = 2
        else:
            args.append(2)
        super().__init__(*args)

    @classmethod
    def dumps(cls, obj, protocol=4):
        return ForkingPickler.dumps(obj, protocol)


def dump(obj, file, protocol=4):
    ForkingPickler4(file, protocol).dump(obj)


class Pickle4Reducer(AbstractReducer):
    ForkingPickler = ForkingPickler4
    register = ForkingPickler4.register
    dump = dump

def strip_punctuations(utterance: str):
    """
    function to strip punctuations from the utternace
    :param utterance:
    :return:
    """
    normalization_pattern = "'s"
    utterance = re.sub(normalization_pattern, " is", utterance)
    puncuation_pattern = "|".join(term_data.PUNCTUATION)
    utterance = re.sub(puncuation_pattern, " ", utterance)
    return utterance

def _preprocess_chi2(workspace_pd):
    """
    Preprocess dataframe for chi2 analysis
    :param workspace_pd: Preprocess dataframe for chi2
    :return labels: intents processed
    :return count_vectorizer: vectorizer instance
    :return features: features from transform
    """
    stopword_list = [strip_punctuations(x) for x in term_data.STOP_WORDS]

    workspace_pd["utterance_punc_stripped"] = workspace_pd["utterance"].apply(
        strip_punctuations
    )

    count_vectorizer = CountVectorizer(
        min_df=1,
        encoding="utf-8",
        ngram_range=(1, 2),
        stop_words=stopword_list,
        tokenizer=word_tokenize,
        token_pattern="(?u)\b\w+\b",
    )
    features = count_vectorizer.fit_transform(
        workspace_pd["utterance_punc_stripped"]
    ).toarray()
    labels = workspace_pd["intent"]
    return labels, count_vectorizer, features


def _compute_chi2_top_feature(
    logger, features, labels, vectorizer, cls, significance_level=0.05
):
    """
    Perform chi2 analysis, punctuation filtering and deduplication
    :param features: count vectorizer features
    :param labels: intents processed
    :param vectorizer: count vectorizer instances
    :param cls: classes for chi square
    :param significance_level: specify an alpha
    :return deduplicated_unigram:
    :return deduplicated_bigram:
    """

    logger.info("Pool calculation agent started for label %s", cls)

    features_chi2, pval = chi2(features, labels == cls)

    logger.info("Chi2 calculated for label %s", cls)

    feature_names = np.array(vectorizer.get_feature_names())

    features_chi2 = features_chi2[pval < significance_level]
    feature_names = feature_names[pval < significance_level]

    indices = np.argsort(features_chi2)
    feature_names = feature_names[indices]

    unigrams = [v.strip() for v in feature_names if len(v.strip().split()) == 1]
    deduplicated_unigram = list()

    for unigram in unigrams:
        if unigram not in deduplicated_unigram:
            deduplicated_unigram.append(unigram)

    bigrams = [v.strip() for v in feature_names if len(v.strip().split()) == 2]

    deduplicated_bigram = list()
    for bigram in bigrams:
        if bigram not in deduplicated_bigram:
            deduplicated_bigram.append(bigram)

    logger.info("compute_chi2_top_feature done for label %s", cls)

    return deduplicated_unigram, deduplicated_bigram

def _compute_chi2_top_feature_obj(obj):
    return _compute_chi2_top_feature(
        obj['logger'], obj['features'], obj['labels'], obj['vectorizer'], obj['label'], obj['significance_level']
    )

def get_chi2_analysis(logger, workspace_pd, num_xgrams=5, significance_level=0.05):
    """
    find correlated unigram and bigram of each intent with Chi2 analysis
    :param workspace_pd: dataframe, workspace data
    :param signficance_level: float, significance value to reject the null hypothesis
    :return unigram_intent_dict:
    :return bigram_intent_dict:
    """
    labels, vectorizer, features = _preprocess_chi2(workspace_pd)

    label_frequency_dict = dict(Counter(workspace_pd["intent"]).most_common())
    N = num_xgrams

    # keys are the set of unigrams/bigrams and value will be the intent
    # maps one-to-many relationship between unigram and intent,
    unigram_intent_dict = dict()
    # maps one-to-many relationship between bigram and intent
    bigram_intent_dict = dict()

    classes = list()
    chi_unigrams = list()
    chi_bigrams = list()
    #manager = mp.Manager()
    #lst = manager.list([])
    #ctx = mp.get_context()
    #ctx.reducer = Pickle4Reducer()
    #pool = mp.Pool(processes=5)
    executer = ThreadPoolExecutor(max_workers = 3)
    #results = executor.map(square, values)
    args = []
    results = []
    for label in label_frequency_dict.keys():
        classes.append(label)
        args.append({
            'features': features,
            'labels': labels,
            'vectorizer': vectorizer,
            'label': label,
            'significance_level': significance_level,
            'logger': logger
        })
        #results.append(_compute_chi2_top_feature(
        #    logger, features, labels, vectorizer, label, significance_level
        #))
    #print('ss')
    results = executer.map(_compute_chi2_top_feature_obj, tuple(args))
    #results = pool.imap(_compute_chi2_top_feature_obj, tuple(args))
    #pool.close()
    #pool.join()
    #print('ss1')

    logger.info("Pool calc done")

    for r in results:
        unigrams, bigrams = r

        if unigrams:
            chi_unigrams.append(unigrams[-N:])
        else:
            chi_unigrams.append([])

        if bigrams:
            chi_bigrams.append(bigrams[-N:])
        else:
            chi_bigrams.append([])

        if unigrams:
            if frozenset(unigrams[-N:]) in unigram_intent_dict:
                unigram_intent_dict[frozenset(unigrams[-N:])].append(label)
            else:
                unigram_intent_dict[frozenset(unigrams[-N:])] = list()
                unigram_intent_dict[frozenset(unigrams[-N:])].append(label)

        if bigrams:
            if frozenset(bigrams[-N:]) in bigram_intent_dict:
                bigram_intent_dict[frozenset(bigrams[-N:])].append(label)
            else:
                bigram_intent_dict[frozenset(bigrams[-N:])] = list()
                bigram_intent_dict[frozenset(bigrams[-N:])].append(label)

    chi_df = [ { 'name': name, 'unigrams': unigrams, 'bigrams': bigrams } for name, unigrams, bigrams in zip(classes, chi_unigrams, chi_bigrams)]

    logger.info("get_chi2_analysis done")

    return chi_df, unigram_intent_dict, bigram_intent_dict

def get_confusing_key_terms(keyterm_intent_map):
    """
    Greedy search for overlapping intents
    :param keyterm_intent_map: correlated terms
    :return df: ambiguous terms data frame
    """
    ambiguous_name1 = list()
    ambiguous_name2 = list()
    ambiguous_keywords = list()
    intents_seen = list()

    for i in range(len(keyterm_intent_map)):
        correlated_unigrams = list(keyterm_intent_map.keys())[i]
        current_label = keyterm_intent_map[correlated_unigrams][0]
        intents_seen.append(current_label)

        for other_correlated_unigrams in keyterm_intent_map.keys():
            other_label = keyterm_intent_map[other_correlated_unigrams][0]
            if other_label in intents_seen:
                continue
            overlap = correlated_unigrams.intersection(other_correlated_unigrams)
            if overlap:
                for keyword in overlap:
                    ambiguous_name1.append(current_label)
                    ambiguous_name2.append(other_label)
                    ambiguous_keywords.append(keyword)

    return [ { 'name1': name1, 'name2': name2, 'keyword': keyword } for name1, name2, keyword in zip(ambiguous_name1, ambiguous_name2, ambiguous_keywords)]

import re
import os
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from api.utils import term_data
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

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
    logger, log_extras, worker_name, features, labels, vectorizer, cls, significance_level=0.05
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

    logger.info("Pool calculation agent started for label %s", cls, extra=log_extras)

    features_chi2, pval = chi2(features, labels == cls)

    logger.info("Chi2 calculated for label %s", cls, extra=log_extras)

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

    logger.info("compute_chi2_top_feature done for label %s", cls, extra=log_extras)

    return deduplicated_unigram, deduplicated_bigram, cls

def _compute_chi2_top_feature_obj(obj):
    return _compute_chi2_top_feature(
        obj['logger'], obj['log_extras'], obj['worker_name'], obj['features'], obj['labels'], obj['vectorizer'], obj['label'], obj['significance_level']
    )

def get_chi2_analysis(logger, log_extras, worker_name, workspace_pd, sendStatus=None, CalcStatus=None, num_xgrams=5, significance_level=0.05):
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
    executer = ThreadPoolExecutor(max_workers = os.environ.get('COACH_THREADS_CHI2_ANALYSIS', 3))
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
            'logger': logger,
            'log_extras': log_extras,
            'worker_name': worker_name
        })
    results = executer.map(_compute_chi2_top_feature_obj, tuple(args))

    data = list()
    for r in results:
        unigrams, bigrams, label = r

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
        
        data.append(r)
        progress = len(data)
        
        if sendStatus is not None and CalcStatus is not None:
            sendStatus('chi2', CalcStatus.CHI2_ANALYSIS_RUNNING,
                    1, 4, f'Chi2 Analysis running  {int(progress * 100/len(args))}% ({progress}/{len(args)})')

    chi_df = [ { 'name': name, 'unigrams': unigrams, 'bigrams': bigrams } for name, unigrams, bigrams in zip(classes, chi_unigrams, chi_bigrams)]

    logger.info("get_chi2_analysis done", extra=log_extras)

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

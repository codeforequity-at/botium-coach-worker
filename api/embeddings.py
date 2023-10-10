from ast import Return
import json
import math
import numpy as np
import openai
import os
import pandas as pd
import pickle
import pinecone
import PyPDF2
import re
import requests
import sys
import time
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_text
import torch

from api.term_analysis import chi2_analyzer, similarity_analyzer
from api.utils import pandas_utils
from enum import Enum
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import current_app
from flask_healthz import healthz
from functools import singledispatch
from joblib import Parallel, delayed
from .redis_client import getRedis
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from .utils.log import getLogger
class CalcStatus(str, Enum):
    CHI2_ANALYSIS_RUNNING = 'CHI2_ANALYSIS_RUNNING'
    CHI2_ANALYSIS_READY = 'CHI2_ANALYSIS_READY'
    CHI2_ANALYSIS_FAILED = 'CHI2_ANALYSIS_FAILED'
    CHI2_AMBIGUOS_UNIGRAMS_RUNNING = 'CHI2_AMBIGUOS_UNIGRAMS_RUNNING'
    CHI2_AMBIGUOS_UNIGRAMS_READY = 'CHI2_AMBIGUOS_UNIGRAMS_READY'
    CHI2_AMBIGUOS_UNIGRAMS_FAILED = 'CHI2_AMBIGUOS_UNIGRAMS_FAILED'
    CHI2_AMBIGUOS_BIGRAMS_RUNNING = 'CHI2_AMBIGUOS_BIGRAMS_RUNNING'
    CHI2_AMBIGUOS_BIGRAMS_READY = 'CHI2_AMBIGUOS_BIGRAMS_READY'
    CHI2_AMBIGUOS_BIGRAMS_FAILED = 'CHI2_AMBIGUOS_BIGRAMS_FAILED'
    CHI2_SIMILARITY_ANALYSIS_RUNNING = 'CHI2_SIMILARITY_ANALYSIS_RUNNING'
    CHI2_SIMILARITY_ANALYSIS_READY = 'CHI2_SIMILARITY_ANALYSIS_READY'
    CHI2_SIMILARITY_ANALYSIS_FAILED = 'CHI2_SIMILARITY_ANALYSIS_FAILED'
    EMBEDDINGS_INTENTS_RUNNING = 'EMBEDDINGS_INTENTS_RUNNING'
    EMBEDDINGS_INTENTS_READY = 'EMBEDDINGS_INTENTS_READY'
    EMBEDDINGS_INTENTS_FAILED = 'EMBEDDINGS_INTENTS_FAILED'
    EMBEDDINGS_PCA_RUNNING = 'EMBEDDINGS_PCA_RUNNING'
    EMBEDDINGS_PCA_READY = 'EMBEDDINGS_PCA_READY'
    EMBEDDINGS_PCA_FAILED = 'EMBEDDINGS_PCA_FAILED'
    EMBEDDINGS_PREPARE_COSINE_SIMILARITY_RUNNING = 'EMBEDDINGS_PREPARE_COSINE_SIMILARITY_RUNNING'
    EMBEDDINGS_PREPARE_COSINE_SIMILARITY_READY = 'EMBEDDINGS_PREPARE_COSINE_SIMILARITY_READY'
    EMBEDDINGS_PREPARE_COSINE_SIMILARITY_FAILED = 'EMBEDDINGS_PREPARE_COSINE_SIMILARITY_FAILED'
    EMBEDDINGS_COSINE_SIMILARITY_RUNNING = 'EMBEDDINGS_COSINE_SIMILARITY_RUNNING'
    EMBEDDINGS_COSINE_SIMILARITY_READY = 'EMBEDDINGS_COSINE_SIMILARITY_READY'
    EMBEDDINGS_COSINE_SIMILARITY_FAILED = 'EMBEDDINGS_COSINE_SIMILARITY_FAILED'


@singledispatch
def to_serializable(val):
    """Used by default."""
    return str(val)

# to generate test data
def objtofile(data, filename, logger):
    with open('test_data/' + filename + '.obj', 'wb') as outfile:
        pickle.dump(data, outfile)


@to_serializable.register(np.float32)
def ts_float32(val):
    """Used if *val* is an instance of numpy.float32."""
    return np.float64(val)


maxUtterancesForEmbeddings = -1
if 'COACH_MAX_UTTERANCES_FOR_EMBEDDINGS' in os.environ:
    maxUtterancesForEmbeddings = int(
        os.environ['COACH_MAX_UTTERANCES_FOR_EMBEDDINGS'])
maxCalcCount = 100
if 'COACH_MAX_CALCULATIONS_PER_WORKER' in os.environ:
    maxCalcCount = int(os.environ['COACH_MAX_CALCULATIONS_PER_WORKER'])


def cosine_similarity_worker(w):
    intent_1 = w[0]
    phrase_1 = w[1]
    embedd_1 = w[2]
    intent_2 = w[3]
    phrase_2 = w[4]
    embedd_2 = w[5]
    similarity = cosine_similarity([embedd_1], [embedd_2])[0][0]
    return [intent_1, phrase_1, intent_2, phrase_2, similarity]


def calculate_embeddings_worker(req_queue, err_queue, processId):
    red = None
    if bool(os.environ.get('REDIS_ENABLE', 0)) == True:
        red = getRedis()
    worker_name = 'Worker ' + str(processId)
    logger = getLogger(worker_name)
    logger.info('Initialize worker ...')
    logger.info('Loading word embeddings model from tfhub ...')
    try:
        generate_embeddings = hub.load(
            'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3')
    except Exception as e:
        err_queue.put(str(e))
    logger.info('Word embeddings model ready.')
    logger.info('Worker started!')
    calc_count = 0
    while calc_count < maxCalcCount:
        embeddingsRequest, method = req_queue.get()
        logger.debug(json.dumps(embeddingsRequest,
                     indent=2, default=to_serializable))
        coachSessionId = embeddingsRequest['coachSessionId'] if 'coachSessionId' in embeddingsRequest else None
        clientId = embeddingsRequest['clientId'] if 'clientId' in embeddingsRequest else None
        testSetId = embeddingsRequest['testSetId'] if 'testSetId' in embeddingsRequest else None
        testSetName = embeddingsRequest['testSetName'] if 'testSetName' in embeddingsRequest else None
        log_extras = {
            "clientId": clientId,
            "testSetId": testSetId,
            "testSetName": testSetName,
            "coachSessionId": coachSessionId
        }
        if 'boxEndpoint' in embeddingsRequest:
            boxEndpoint = embeddingsRequest['boxEndpoint']
        else:
            boxEndpoint = None
        filter = embeddingsRequest['filter'] if 'filter' in embeddingsRequest else None
        intents = embeddingsRequest['intents'] if 'intents' in embeddingsRequest else None

        logger.info('Filter is %s', filter, extra=log_extras)

        def sendStatus(category, calc_status, step, max_steps, message):
            logger.info(message, extra=log_extras)
            if bool(os.environ.get('REDIS_ENABLE', False)) == True:
                logger.info('Sending "%s" to redis', message, extra=log_extras)
                red.set('coachworker_status_' + category + '_' + coachSessionId, json.dumps({
                    "method": "calculate_" + category,
                    "clientId": clientId,
                    "coachSessionId": coachSessionId,
                    "status": calc_status,
                    "statusDescription": message,
                    "step": step,
                    "steps": max_steps
                }), ex=600)

        # for testing purposes on local environment
        if 'COACH_DEV_BOX_ENDPOINT' in os.environ:
            boxEndpoint = os.environ.get('COACH_DEV_BOX_ENDPOINT')
        if method == "retryRequest":
            seconds = int(os.environ.get('COACH_RETRY_REQUEST_DELAY', 10))
            max_retries = int(os.environ.get(
                'COACH_RETRY_REQUEST_RETRIES', 12))
            if 'retry' in embeddingsRequest:
                retry = int(embeddingsRequest["retry"]) + 1
            else:
                retry = 1
            logger.info('next retry request for %s ( %s of %s ) in %s seconds',
                        embeddingsRequest["retry_method"],
                        retry,
                        max_retries,
                        seconds,
                        extra=log_extras
                        )
            time.sleep(seconds)
            try:
                if 'header' in embeddingsRequest.keys():
                    if 'json' in embeddingsRequest.keys():
                        res = requests.post(
                            boxEndpoint, headers=embeddingsRequest["header"], json=embeddingsRequest["json"])
                    else:
                        res = requests.post(
                            boxEndpoint, headers=embeddingsRequest["header"], data=embeddingsRequest["data"])
                else:
                    if 'json' in embeddingsRequest.keys():
                        res = requests.post(
                            boxEndpoint, json=embeddingsRequest["json"])
                    else:
                        res = requests.post(
                            boxEndpoint, data=embeddingsRequest["data"])
                if res.status_code != 200:
                    raise Exception('Wrong status code ' +
                                    str(res.status_code))
                logger.info(str(res), extra=log_extras)
                logger.info('retry request for %s ( %s of %s ) to %s successfully sent',
                            embeddingsRequest["retry_method"],
                            retry,
                            max_retries,
                            boxEndpoint,
                            extra=log_extras)
            except Exception as e:
                logger.info('%s', e, extra=log_extras)
                if retry <= max_retries:
                    logger.info('retry request for %s ( %s of %s ) to %s failed, trying again',
                                embeddingsRequest["retry_method"],
                                retry,
                                max_retries,
                                boxEndpoint,
                                extra=log_extras)
                    retry_request = {
                        "retry": retry,
                        "boxEndpoint": boxEndpoint,
                        "retry_method": embeddingsRequest['retry_method']
                    }
                    if 'json' in embeddingsRequest:
                        retry_request["json"] = embeddingsRequest['json']
                    else:
                        retry_request["data"] = embeddingsRequest['data']
                    if 'header' in embeddingsRequest:
                        retry_request["header"] = embeddingsRequest['header']
                    req_queue.put((retry_request, "retryRequest"))
                else:
                    logger.info('retry request for %s ( %s of %s ) to %s failed, no tries anymore',
                                embeddingsRequest["retry_method"],
                                retry,
                                max_retries,
                                boxEndpoint,
                                extra=log_extras
                                )
        if method == "calculate_chi2":
            try:
                if len(intents) == 0:
                    response_data = {
                        "method": "calculate_chi2",
                        "status": "finished",
                        "coachSessionId": coachSessionId,
                        "clientId": clientId,
                        "testSetId": testSetId,
                        "testSetName": testSetName,
                        "output": {
                            'chi2': [],
                            'chi2_ambiguous_unigrams': [],
                            'chi2_ambiguous_bigrams': [],
                            'chi2_similarity': []
                        }
                    }
                    logger.debug(json.dumps(
                        response_data, indent=2), extra=log_extras)
                    if boxEndpoint is not None:
                        try:
                            res = requests.post(
                                boxEndpoint, json=response_data)
                            if res.status_code != 200:
                                raise Exception(
                                    'Wrong status code ' + str(res.status_code))
                            logger.info(str(res), extra=log_extras)
                        except Exception as e:
                            logger.error('Sending chi2 failed: ' +
                                         str(e), extra=log_extras)
                            req_queue.put(({
                                "boxEndpoint": boxEndpoint,
                                "json": response_data,
                                "retry_method": "calculate_chi2"
                            }, "retryRequest"))
                    else:
                        red.set('coachworker_res_chi2_' +
                                coachSessionId, json.dumps(response_data), ex=3600)
                        red.delete('coachworker_req_chi2_' + coachSessionId)
                    continue

                if not 'maxxgrams' in filter:
                    filter['maxxgrams'] = 5

                flattenedForChi2 = pandas_utils.flatten_intents_list(intents)
                #objtofile(flattenedForChi2, 'flattenedForChi2', logger)

                sendStatus('chi2', CalcStatus.CHI2_ANALYSIS_RUNNING, 1, 4,
                           'Chi2 Analysis running')
                try:
                    chi2, unigram_intent_dict, bigram_intent_dict = chi2_analyzer.get_chi2_analysis(
                        logger, log_extras, worker_name, flattenedForChi2, num_xgrams=filter['maxxgrams'])
                    #objtofile(chi2, 'chi2', logger)
                    #objtofile(unigram_intent_dict, 'unigram_intent_dict', logger)
                    #objtofile(bigram_intent_dict, 'bigram_intent_dict', logger)
                except Exception as e:
                    sendStatus('chi2', CalcStatus.CHI2_ANALYSIS_FAILED, 1, 4,
                               'Chi2 analysis failed - {}'.format(e))
                sendStatus('chi2', CalcStatus.CHI2_ANALYSIS_READY, 1, 4,
                           'Chi2 analysis ready')

                sendStatus(
                    'chi2', CalcStatus.CHI2_AMBIGUOS_UNIGRAMS_RUNNING, 2, 4, 'Chi2 Ambiguous Unigrams Analysis running')
                try:
                    chi2_ambiguous_unigrams = chi2_analyzer.get_confusing_key_terms(
                        unigram_intent_dict)
                    #objtofile(chi2_ambiguous_unigrams, 'chi2_ambiguous_unigrams', logger)
                except Exception as e:
                    sendStatus('chi2', CalcStatus.CHI2_AMBIGUOS_UNIGRAMS_FAILED, 2, 4,
                               'Chi2 Ambiguous Unigrams Analysis failed - {}'.format(e))
                sendStatus(
                    'chi2', CalcStatus.CHI2_AMBIGUOS_UNIGRAMS_READY, 2, 4, 'Chi2 Ambiguous Unigrams Analysis ready')

                sendStatus(
                    'chi2', CalcStatus.CHI2_AMBIGUOS_BIGRAMS_RUNNING, 3, 4, 'Chi2 Ambiguous Bigrams Analysis running')
                try:
                    chi2_ambiguous_bigrams = chi2_analyzer.get_confusing_key_terms(
                        bigram_intent_dict)
                    #objtofile(chi2_ambiguous_bigrams, 'chi2_ambiguous_bigrams', logger)
                except Exception as e:
                    sendStatus(
                        'chi2', CalcStatus.CHI2_AMBIGUOS_BIGRAMS_FAILED, 3, 4, 'Chi2 Ambiguous Bigrams Analysis failed - {}'.format(e))
                sendStatus(
                    'chi2', CalcStatus.CHI2_AMBIGUOS_BIGRAMS_READY, 3, 4, 'Chi2 Ambiguous Bigrams Analysis ready')

                sendStatus(
                    'chi2', CalcStatus.CHI2_SIMILARITY_ANALYSIS_RUNNING, 4, 4, 'Chi2 Similarity Analysis running')
                try:
                    chi2_similarity = similarity_analyzer.ambiguous_examples_analysis(
                        logger, log_extras, worker_name, flattenedForChi2, filter['minsimilarity'])
                    #objtofile(chi2_similarity, 'chi2_similarity', logger)
                except Exception as e:
                    sendStatus(
                        'chi2', CalcStatus.CHI2_SIMILARITY_ANALYSIS_FAILED, 4, 4, 'Chi2 Similarity Analysis failed - {}'.format(e))
                sendStatus(
                    'chi2', CalcStatus.CHI2_SIMILARITY_ANALYSIS_READY, 4, 4, 'Chi2 Similarity Analysis ready')
                logger.info('Returning results', extra=log_extras)

                logger.info('Sending results to %s',
                            boxEndpoint, extra=log_extras)
                response_data = {
                    "method": "calculate_chi2",
                    "status": "finished",
                    "coachSessionId": coachSessionId,
                    "clientId": clientId,
                    "testSetId": testSetId,
                    "testSetName": testSetName,
                    "output": {
                        'chi2': chi2,
                        'chi2_ambiguous_unigrams': chi2_ambiguous_unigrams,
                        'chi2_ambiguous_bigrams': chi2_ambiguous_bigrams,
                        'chi2_similarity': chi2_similarity
                    }
                }
                if boxEndpoint is not None:
                    header = {"content-type": "application/json"}
                    data = json.dumps(response_data, default=to_serializable)
                    try:
                        res = requests.post(
                            boxEndpoint, headers=header, data=data)
                        if res.status_code != 200:
                            raise Exception(
                                'Wrong status code ' + str(res.status_code))
                        logger.info(str(res), extra=log_extras)
                    except Exception as e:
                        logger.error('Sending chi2 failed: ' +
                                     str(e), extra=log_extras)
                        req_queue.put(({
                            "boxEndpoint": boxEndpoint,
                            "header": header,
                            "data": data,
                            "retry_method": "calculate_chi2"
                        }, "retryRequest"))
                else:
                    data = json.dumps(response_data, default=to_serializable)
                    red.set('coachworker_res_chi2_' + coachSessionId, data)
                    red.delete('coachworker_req_chi2_' + coachSessionId)
                calc_count += 1
            except Exception as e:
                logger.error('Calculating chi2 failed: ' +
                             str(e), extra=log_extras)
                response_data = {
                    "method": "calculate_chi2",
                    "status": "failed",
                    "coachSessionId": coachSessionId,
                    "clientId": clientId,
                    "testSetId": testSetId,
                    "testSetName": testSetName,
                    "err": 'Calculating chi2 failed: ' + str(e)
                }
                logger.debug(json.dumps(response_data, indent=2))
                try:
                    res = requests.post(boxEndpoint, json=response_data)
                    if res.status_code != 200:
                        raise Exception('Wrong status code ' +
                                        str(res.status_code))
                    logger.info('%s: ' + str(res), worker_name)
                except Exception as e:
                    logger.error('Sending chi2 failed: ' +
                                 str(e), extra=log_extras)
                    req_queue.put(({
                        "boxEndpoint": boxEndpoint,
                        "json": response_data,
                        "retry_method": "calculate_chi2"
                    }, "retryRequest"))
        if method == "calculate_embeddings":
            try:
                if len(intents) == 0:
                    response_data = {
                        "method": "calculate_embeddings",
                        "status": "finished",
                        "coachSessionId": coachSessionId,
                        "clientId": clientId,
                        "testSetId": testSetId,
                        "testSetName": testSetName,
                        "output": {
                            'embeddings': embeddings_coords,
                            'similarity': similarity,
                            'cohesion': cohesion,
                            'separation': separation,
                        }
                    }
                    logger.debug(json.dumps(
                        response_data, indent=2), extra=log_extras)
                    if boxEndpoint is not None:
                        try:
                            res = requests.post(
                                boxEndpoint, json=response_data)
                            if res.status_code != 200:
                                raise Exception(
                                    'Wrong status code ' + str(res.status_code))
                            logger.info(str(res), extra=log_extras)
                        except Exception as e:
                            logger.error(
                                'Sending embeddings failed: ' + str(e), extra=log_extras)
                            req_queue.put(({
                                "boxEndpoint": boxEndpoint,
                                "json": response_data,
                                "retry_method": "calculate_embeddings"
                            }, "retryRequest"))
                    else:
                        red.set('coachworker_res_embeddings_' +
                                coachSessionId, json.dumps(response_data))
                        red.delete('coachworker_req_embeddings_' +
                                   coachSessionId)
                    continue

                if not 'maxxgrams' in filter:
                    filter['maxxgrams'] = 5

                sendStatus(
                    'embeddings', CalcStatus.EMBEDDINGS_INTENTS_RUNNING, 1, 4, "Embeddings calculation for {} intents running".format(len(intents)))
                try:
                    for intent in intents:
                        logger.info('Calculating embeddings for intent "%s" with %s: examples',
                                    intent['name'], len(intent['examples']), extra=log_extras)

                    training_phrases_with_embeddings = defaultdict(list)
                    for intent in intents:
                        if len(intent['examples']) > 0:
                            computed_embeddings = generate_embeddings(
                                intent['examples'])
                            training_phrases_with_embeddings[intent['name']] = dict(
                                zip(intent['examples'], computed_embeddings))

                    for intent_name, _ in training_phrases_with_embeddings.items():
                        training_phrase, embeddings = next(
                            iter(training_phrases_with_embeddings[intent_name].items()))
                        logger.info('Calculated embeddings for intent {}, example: {{\'{}\':{}}}'.format(
                            intent_name, training_phrase, embeddings[:5]), extra=log_extras)

                    embedding_vectors = []

                    for intent, training_phrases_and_embeddings in training_phrases_with_embeddings.items():
                        for training_phrase, embeddings in training_phrases_and_embeddings.items():
                            embedding_vectors.append(embeddings)

                    embedding_vectors = np.asarray(embedding_vectors)
                except Exception as e:
                    sendStatus(
                        'embeddings', CalcStatus.EMBEDDINGS_INTENTS_FAILED, 1, 4, "Embeddings calculation for {} intents failed - {}".format(len(intents), e))
                    exit(1)

                sendStatus(
                    'embeddings', CalcStatus.EMBEDDINGS_INTENTS_READY, 1, 4, "Embeddings calculation for {} intents ready".format(len(intents)))

                sendStatus(
                    'embeddings', CalcStatus.EMBEDDINGS_PCA_RUNNING, 2, 4, 'Principal Component Analysis running')
                try:
                    logger.info('Starting principal component analysis for %s examples', len(
                        embedding_vectors), extra=log_extras)

                    pca = PCA(n_components=2)
                    pca.fit(embedding_vectors)

                    embeddings_coords = []

                    for color, intent in enumerate(training_phrases_with_embeddings.keys()):
                        phrases = list(
                            training_phrases_with_embeddings[intent].keys())
                        embeddings = list(
                            training_phrases_with_embeddings[intent].values())
                        points = pca.transform(embeddings)
                        embeddings_coords.append({
                            'name': intent,
                            'examples': [{'phrase': phrase, 'x': np.float(point[0]), 'y': np.float(point[1])} for phrase, point in zip(phrases, points)]
                        })

                    logger.debug(json.dumps(embeddings_coords, indent=2))
                    logger.info('Ready with principal component analysis for %s examples', len(
                        embedding_vectors), extra=log_extras)

                    flattenedForCosine = []

                    for intent in training_phrases_with_embeddings:
                        phrases = list(
                            training_phrases_with_embeddings[intent].keys())
                        if maxUtterancesForEmbeddings > 0:
                            utterancesForIntent = math.ceil(
                                len(phrases) * maxUtterancesForEmbeddings / len(embedding_vectors))
                            if utterancesForIntent < len(phrases):
                                logger.info('Randomly selecting %s: examples for intent %s: for cosine similarity',
                                            utterancesForIntent, intent, extra=log_extras)
                                phrases = np.random.choice(
                                    phrases, utterancesForIntent, replace=False)
                        for phrase in phrases:
                            flattenedForCosine.append(
                                (intent, phrase, training_phrases_with_embeddings[intent][phrase]))
                except Exception as e:
                    sendStatus(
                        'embeddings', CalcStatus.EMBEDDINGS_PCA_FAILED, 2, 4, 'Principal Component Analysis failed - {}'.format(e))
                    exit(1)
                sendStatus(
                    'embeddings', CalcStatus.EMBEDDINGS_PCA_READY, 2, 4, 'Principal Component Analysis ready')

                sendStatus(
                    'embeddings', CalcStatus.EMBEDDINGS_PREPARE_COSINE_SIMILARITY_RUNNING, 3, 4, 'Preparation for Cosine Similarity Analysis running')
                try:
                    logger.info('Preparing cosine similarity for %s examples', len(
                        flattenedForCosine), extra=log_extras)

                    workers = []
                    for i in range(len(flattenedForCosine)):
                        for j in range(i+1, len(flattenedForCosine)):
                            intent_1 = flattenedForCosine[i][0]
                            phrase_1 = flattenedForCosine[i][1]
                            embedd_1 = flattenedForCosine[i][2]

                            intent_2 = flattenedForCosine[j][0]
                            phrase_2 = flattenedForCosine[j][1]
                            embedd_2 = flattenedForCosine[j][2]

                            workers.append(
                                (intent_1, phrase_1, embedd_1, intent_2, phrase_2, embedd_2))
                except Exception as e:
                    sendStatus(
                        'embeddings', CalcStatus.EMBEDDINGS_PREPARE_COSINE_SIMILARITY_FAILED, 3, 4, 'Preparation for Cosine Similarity Analysis failed - {}'.format(e))
                    exit(1)

                sendStatus(
                    'embeddings', CalcStatus.EMBEDDINGS_PREPARE_COSINE_SIMILARITY_READY, 3, 4, 'Preparation for Cosine Similarity Analysis ready')

                sendStatus(
                    'embeddings', CalcStatus.EMBEDDINGS_COSINE_SIMILARITY_RUNNING, 4, 4, 'Cosine Similarity Analysis running')
                logger.info('Running cosine similarity for %s examples', len(
                    flattenedForCosine), extra=log_extras)

                # data = Parallel(n_jobs=-1)(delayed(cosine_similarity_worker)(w[0], w[1], w[2], w[3], w[4], w[5]) for w in workers)
                executer = ThreadPoolExecutor(max_workers=os.environ.get(
                    'COACH_THREADS_EMBEDDINGS_COSINE_SIMILARITY', 3))
                data = list(executer.map(
                    cosine_similarity_worker, tuple(workers)))

                logger.info('Ready with cosine similarity for %s pairs, preparing results', len(
                    data), extra=log_extras)

                similarity_df = pd.DataFrame(
                    data, columns=['name1', 'example1', 'name2', 'example2', 'similarity'])
                similarity_different_intent = similarity_df['name1'] != similarity_df['name2']
                similarity_same_intent = similarity_df['name1'] == similarity_df['name2']

                similarity_different_intent_filtered = (similarity_df['name1'] != similarity_df['name2']) & (
                    similarity_df['similarity'] > filter['minsimilarity'])
                similarity_df_sorted = similarity_df[similarity_different_intent_filtered].sort_values(
                    'similarity', ascending=False)
                similarity = [{'name1': name1, 'example1': example1, 'name2': name2, 'example2': example2, 'similarity': similarity} for name1, example1, name2, example2, similarity in zip(
                    similarity_df_sorted['name1'], similarity_df_sorted['example1'], similarity_df_sorted['name2'], similarity_df_sorted['example2'], similarity_df_sorted['similarity'])]
                logger.debug(json.dumps(similarity, indent=2),
                             extra=log_extras)

                cohesion_df_sorted = pd.DataFrame(similarity_df[similarity_same_intent].groupby(
                    'name1', as_index=False)['similarity'].mean()).sort_values('similarity', ascending=False)
                cohesion_df_sorted.columns = ['name', 'cohesion']
                cohesion = [{'name': name, 'cohesion': cohesion} for name, cohesion in zip(
                    cohesion_df_sorted['name'], cohesion_df_sorted['cohesion'])]
                logger.debug(json.dumps(cohesion, indent=2), extra=log_extras)

                separation_df_sorted = pd.DataFrame(similarity_df[similarity_different_intent].groupby(
                    ['name1', 'name2'], as_index=False)['similarity'].mean()).sort_values('similarity', ascending=True)
                separation_df_sorted['separation'] = 1 - \
                    separation_df_sorted['similarity']
                separation = [{'name1': name1, 'name2': name2, 'separation': separation} for name1, name2, separation in zip(
                    separation_df_sorted['name1'], separation_df_sorted['name2'], separation_df_sorted['separation'])]
                logger.debug(json.dumps(separation, indent=2),
                             extra=log_extras)
                sendStatus(
                    'embeddings', CalcStatus.EMBEDDINGS_COSINE_SIMILARITY_READY, 4, 4, 'Cosine Similarity Analysis ready')

                logger.info('Returning results', extra=log_extras)

                logger.info('Sending results to %s',
                            boxEndpoint, extra=log_extras)
                response_data = {
                    "method": "calculate_embeddings",
                    "status": "finished",
                    "coachSessionId": coachSessionId,
                    "clientId": clientId,
                    "testSetId": testSetId,
                    "testSetName": testSetName,
                    "output": {
                        'embeddings': embeddings_coords,
                        'similarity': similarity,
                        'cohesion': cohesion,
                        'separation': separation
                    }
                }
                logger.debug(json.dumps(response_data, indent=2),
                             extra=log_extras)
                if boxEndpoint is not None:
                    try:
                        res = requests.post(boxEndpoint, json=response_data)
                        if res.status_code != 200:
                            raise Exception(
                                'Wrong status code ' + str(res.status_code))
                        logger.info(str(res), extra=log_extras)
                    except Exception as e:
                        logger.error('Sending embeddings failed: ' +
                                     str(e), extra=log_extras)
                        req_queue.put(({
                            "boxEndpoint": boxEndpoint,
                            "json": response_data,
                            "retry_method": "calculate_embeddings"
                        }, "retryRequest"))
                else:
                    red.set('coachworker_res_embeddings_' +
                            coachSessionId, json.dumps(response_data), ex=3600)
                    red.delete('coachworker_req_embeddings_' + coachSessionId)
                calc_count += 1
            except Exception as e:
                logger.error('Calculating embeddings failed: ' +
                             str(e), extra=log_extras)
                response_data = {
                    "method": "calculate_embeddings",
                    "status": "failed",
                    "coachSessionId": coachSessionId,
                    "err": 'Calculating embeddings failed: ' + str(e)
                }
                logger.debug(json.dumps(response_data, indent=2))
                try:
                    res = requests.post(boxEndpoint, json=response_data)
                    if res.status_code != 200:
                        raise Exception('Wrong status code ' +
                                        str(res.status_code))
                    logger.info(str(res), extra=log_extras)
                except Exception as e:
                    logger.error('Sending embeddings failed: ' +
                                 str(e), extra=log_extras)
                    req_queue.put(({
                        "boxEndpoint": boxEndpoint,
                        "json": response_data,
                        "retry_method": "calculate_embeddings"
                    }, "retryRequest"))


def ping():
    return 'Botium Coach Worker. Tensorflow Version: {tfVersion} PyTorch Version: {ptVersion}, Cuda: {ptCuda}'.format(
        tfVersion=tf.__version__, ptVersion=torch.__version__, ptCuda=str(torch.cuda.is_available()))


def calculate_embeddings(embeddingsRequest):
    coachSessionId = embeddingsRequest['coachSessionId'] if 'coachSessionId' in embeddingsRequest else None
    clientId = embeddingsRequest['clientId'] if 'clientId' in embeddingsRequest else None
    testSetId = embeddingsRequest['testSetId'] if 'testSetId' in embeddingsRequest else None
    testSetName = embeddingsRequest['testSetName'] if 'testSetName' in embeddingsRequest else None
    boxEndpoint = embeddingsRequest['boxEndpoint']
    if 'COACH_DEV_BOX_ENDPOINT' in os.environ:
        boxEndpoint = os.environ.get('COACH_DEV_BOX_ENDPOINT')

    try:
        print('Checking callback url availability (' + boxEndpoint + ') ...')
        response_data = {
            "method": "ping"
        }
        res = requests.post(boxEndpoint, json=response_data)
        if res.status_code != 200 and res.status_code != 400:
            raise Exception(
                'Ping check for callback url failed: Status Code ' + str(res.status_code))
    except Exception as e:
        print('Error: Checking callback url availability: ' + str(e))
        return {
            'status': 'rejected',
            'coachSessionId': coachSessionId,
            "clientId": clientId,
            "testSetId": testSetId,
            "testSetName": testSetName,
            'boxEndpoint': boxEndpoint,
            'workerEndpoint': os.environ.get('COACH_HOSTNAME', ''),
            'error_message': str(e)
        }

    with current_app.app_context():
        req_queue = current_app.req_queue
        req_queue.put((embeddingsRequest, "calculate_chi2"))
        req_queue.put((embeddingsRequest, "calculate_embeddings"))

    return {
        'status': 'queued',
        'coachSessionId': coachSessionId,
        "clientId": clientId,
        "testSetId": testSetId,
        "testSetName": testSetName,
        'boxEndpoint': boxEndpoint,
        'workerEndpoint': os.environ.get('COACH_HOSTNAME', '')
    }
#####################################################################################################
#                       Fact checking functions - added by Brian M  9th Oct 2023                    #
#####################################################################################################

def create_query(openai,response_llm):
    """ Create query/facts which are required to be verified

        Input:  openai (class) - OpenAI API client 
                response_llm (string) - statement to be fact checked

        Output: questions (List) - list of questions to gather evidence to fact check statement
    """
    response_llm='Statement:= '+ response_llm
    print(response_llm)
    # Creating queries from the statement 
    print('-------------------------------------------------------------------------------------------------------')
    response=openai.ChatCompletion.create(
      model="gpt-3.5-turbo",messages=[
            {"role": "system", "content": "You are a helpful assistant with the ability to verify the facts in a given statement. Your task is to read the provided statement and break it down into individual facts, sentences, or contexts that require verification. Each aspect of the statement should be treated with a level of skepticism, assuming that there might be some factual errors. Your role is to generate queries to validate each fact, seeking clarification to ensure accurate and consistent information. Please assist in fact-checking by asking questions to verify the details presented in the statement."},
            {"role": "user", "content": "Statement:= Time of My Life is a song by American singer-songwriter Bill Medley from the soundtrack of the 1987 film Dirty Dancing. The song was produced by Michael Lloyd"},
            {"role": "assistant", "content": "Query generation \n Verify:= 1.Who sings the song Time of My Life? \n Verify:= 2.Is the song writer American?\n Verify:= 3.Which year the song was sung?\n Verify:= 4.Which film is the song Time of My Life from? \n Verify:= 5.Who produced the song Time of My Life?"},
            {"role": "user", "content": "Statement:= Your nose switches back and forth between nostrils. When you sleep, you switch about every 45 minutes. This is to prevent a buildup of mucus. It’s called the nasal cycle."},
            {"role": "assistant", "content": "Query generation \n Verify:= 1.Does your nose switch between nostrils? \n Verify:= 2.How often does your nostrils switch? \n Verify:= 3.Why does your nostril switch? \n Verify:= 4.What is nasal cycle?"},
          {"role": "user", "content":response_llm }
        ]
        )
    
    api_response=response['choices'][0]['message']['content']
    questions = []
    search_string='Verify'
    for question in api_response.split("\n"):
            # Remove the search string from each question
        if search_string not in question:
            continue
        question = question.split(search_string)[1].strip()
        questions.append(question)
        
    return questions
# gets context passages from the pinecone index
def get_context(question, index, indexname, top_k):
    """ Generate embeddings for the question

        Input:  question (string) - question used to gather evidence on statement from documents stored on pinecone
                index (class) - Pinecone API client
                indexname (string) - name of index used to store embeddings in pinecone
                top_k (int) - sets The number of results to return for each query

        Output: context (dict) - returns most relevant contect based on questions asked and retrieval score from pineocne
    """
    result = openai.Embedding.create(model="text-embedding-ada-002",input=question)
    embedding=result["data"][0]["embedding"]
    # search pinecone index for context passage with the answer
    context = index.query(namespace=indexname, vector = embedding, top_k=top_k, include_metadata=True)
    return context


# For each question retrieve the most relvant part of document 
def retrieval_passage(openai, response_llm, pineindex, indexname) :
    """ Generate embeddings for the question

        Input:  openai (class) - OpenAI API client 
                response_llms (string) - statement to be fact checked
                pineindex (class) - Pinecone API client
                indexname (string) - name of index specified by user to be created


        Output: used_evidences (list) - list of evidence to support if statement is true or false
    """
    questions=create_query(openai,response_llm)
    used_evidences=[]
    if len(questions)==0:
        return used_evidences
    
    print('Queries Created from the statement',questions)
    print('-------------------------------------------------------------------------------------------------------')
    query_search = []
    for query in questions:
        print('Retrieving relevant passage for query:', query )
        retrieved_passages = []
        #gets context passages from the pinecone index
        context = get_context(query, pineindex, indexname, top_k=1)

        for passage in context["matches"]:
               retrieved_passages.append(
                   {
                       "text": passage['metadata']['context'],
                       "query": query,
                       "retrieval_score": passage['score']
                   }
               )

        print(retrieved_passages) 
        ## figure conflicting articles and stop 
        if retrieved_passages:
            # Sort all retrieved passages by the retrieval score.
            retrieved_passages = sorted(
                retrieved_passages, key=lambda d: d["retrieval_score"], reverse=True
            )

            # Normalize the retreival scores into probabilities
            scores = [r["retrieval_score"] for r in retrieved_passages]
            probs = torch.nn.functional.softmax(torch.Tensor(scores), dim=-1).tolist()
            for prob, passage in zip(probs, retrieved_passages):
                passage["score"] = prob
        query_search.append(retrieved_passages)
        
    used_evidences=[e for cur_evids in query_search for e in cur_evids[:1]]
    return used_evidences

def agreement_gate(openai,response_llm,pineindex, indexname):
    """ Generate embeddings for the question

        Input:  openai (class) - OpenAI API client 
                response_llms (string) - statement to be fact checked
                pineindex (class) - Pinecone API client
                indexname (string) - name of index specified by user to be created

        Output: agreement_gates (list) - contains reasoning, decision and is_open flag of query based on statemment  
                used_evidences (list) - list of evidence to support if statment is true or false
                relevance (int) - shows relevance of passage 
    """

    #Calling retrieval stage before agreement stage
    used_evidences=retrieval_passage(openai,response_llm, pineindex, indexname)
    agreement_responses=[]
    agreement_gates=[]
    
    # Checking relevant articles are present or not in the dataset provided
    relevance=0
    
    print('\n')
    print('-------------------------------------------------------------------------------------------------------')
    print('Evidences gathered for each query we are fact checking ')
    print('*************************************************')
    for i in used_evidences:
        print(i)
        print('*************************************************')
    print('-------------------------------------------------------------------------------------------------------')
    print('\n')
    # No evidence then return empty
    if len(used_evidences)==0:
        return agreement_gates,used_evidences,relevance
    
    for i in range(len(used_evidences)):
        if used_evidences[i]['retrieval_score']<0:
            relevance+=1
            
    if relevance >0:
        return agreement_gates,used_evidences,relevance
    
    for i in range(len(used_evidences)):
        user_llm=  "Statement:= " + response_llm + " \n Query:= " + used_evidences[i]['query'] + " \n Article:= " + used_evidences[i]['text']
        response=openai.ChatCompletion.create(
                  model="gpt-4",
                  messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in performing fact-checking between a given statement and an accompanying document based on the queries provided. Your goal is to ensure consistent and accurate results throughout the fact-checking process. For each query, you will compare both the statement and the document to determine if they agree or disagree on the specific facts presented. Any even slight agreement or disagreement between the two will be concluded as disagree. You will thoroughly provide reasoning for each conclusion reached and in therefore explicilty tell if you agree or disagree. If there are any discrepancies or inconsistencies between the statement and the article you will explicitly state disagree for clarity."  },
                {"role": "user", "content": "Statement:= Time of My Life is a song by American singer-songwriter Bill Medley from the soundtrack of the 1987 film Dirty Dancing. The song was produced by Michael Lloyd. \n Query:= Who was the producer of (I’ve Had) The Time of My Life? \n Article:= On September 8, 2010, the original demo of this song, along with a remix by producer Michael Lloyd , was released as digital files in an effort to raise money for the Patrick Swayze Pancreas Cancer Resarch Foundation at Stanford University."},
                {"role": "assistant", "content": "Reasoning:= The article said that a demo was produced by Michael Lloyd and you said Time of My Life was produced by Michael Lloyd. \n Therefore:= This agrees with statement claims."},
                {"role": "user", "content": "Statement:= Your nose switches back and forth between nostrils. When you sleep, you switch about every 45 minutes. This is to prevent a buildup of mucus. It’s called the nasal cycle. \n Query:= How often do your nostrils switch? \n Article:= Although we don’t usually notice it, during the nasal cycle one nostril becomes congested and thus contributes less to airflow, while the other becomes decongested. On average, the congestion pattern switches about every 2 hours, according to a small 2016 study published in the journal PLOS One."},
                {"role": "assistant", "content": "Reasoning:= The article said the nose’s switching time is about every 2 hours, and you said the nose's switching time is about every 45 minutes. \n Therefore:= This disagrees with statement claims."},
                {"role": "user", "content":user_llm }])
        agreement_responses.append(response)
        
    
    for i in range(len(agreement_responses)):
        api_response=agreement_responses[i]['choices'][0]['message']['content'].strip().split("\n")
        if len(api_response)<2:
            reason = "Failed to parse."
            decision = None
            is_open = False
        else:
            reason = api_response[0]
            decision = api_response[1].split("Therefore:")[-1].strip()
            is_open = "disagrees" in api_response[1]
            gate = {"is_open": is_open, "reason": reason, "decision": decision}
            agreement_gates.append(gate)
    return agreement_gates,used_evidences,relevance


def editor(openai,response_llm, pineindex, indexname):
    """
    Create the Pinecone index

    Inputs: openai (class) - OpenAI API client 
            response_llms (string) - statement to be fact checked
            pineindex (string) - region where index is to be stored
            indexname (string) - name of index specified by user to be created

    Output: edited_response (string) - statement if edited ot not
            agreeemnet_gate (dict) - contains reason, decisions and gate value 
            status (boolean) - returns True if statement has passed fact check  
    """
    agreement_gates,used_evidences,relevance=agreement_gate(openai,response_llm, pineindex, indexname)
    edit_count=0
    edited_responses=[]
    
    if len(agreement_gates)==0 and len(used_evidences)==0 and relevance==0:
        print('Not enough data in the statement for performing fact checking')
        return edited_responses
    
    if relevance == len(used_evidences):
        print('There is no document which is relevant to any/some of the facts present in statement')
        return edited_responses
    
    
    print('-------------------------------------------------------------------------------------------------------')
    print('Agreement gate for each query if the statement agrees or not')
    print('*************************************************')
    for i in agreement_gates:
        print(i)
        print('*************************************************')
    print('-------------------------------------------------------------------------------------------------------')
    print('\n')
    for i in range(len(agreement_gates)):
        if agreement_gates[i]['is_open']:
            user_llm=  "Statement:= " + response_llm + " \n Query:= " + used_evidences[i]['query'] + " \n Article:= " + used_evidences[i]['text'] + agreement_gates[i]['reason']
            response=openai.ChatCompletion.create(
                          model="gpt-4",
                          messages=[
                        #{"role": "system", "content": "You are a helpful assistant.Who fixes the statement using the reasoning provided as there is a disagreement between article and statement on the query."  },
                        {"role": "system", "content": "You are a helpful assistant specializing in fixing statements based on the provided reasoning when there is a disagreement between the statement and the provided documentation on the given query. Your objective is to ensure consistent and accurate results by modifying the facts in the statement using information from the accompanying documentation, guided by the understanding provided in the reasoning."  },#Just modify the facts don't try to add any new information, try to keep as close as possible to the original statement
                        {"role": "user", "content": "Statement:= Time of My Life is a song by American singer-songwriter Bill Medley from the soundtrack of the 1987 film Dirty Dancing. The song was produced by Michael Lloyd. \n Query:= Who was the producer of (I’ve Had) The Time of My Life? \n Article:= On September 8, 2010, the original demo of this song, along with a remix by producer Michael Lloyd , was released as digital files in an effort to raise money for the Patrick Swayze Pancreas Cancer Resarch Foundation at Stanford University. \n Reasoning:= Time of My Life producer name in your statement is wrong."},
                        {"role": "assistant", "content": "Fixed Statement:= Time of My Life is a song by American singer-songwriter Bill Medley from the soundtrack of the 1987 film Dirty Dancing. The song was produced by Michael Lloyd."},
                        {"role": "user", "content": "Statement:= Your nose switches back and forth between nostrils. When you sleep, you switch about every 45 minutes. This is to prevent a buildup of mucus. It’s called the nasal cycle. \n Query:= How often do your nostrils switch? \n Article:= Although we don’t usually notice it, during the nasal cycle one nostril becomes congested and thus contributes less to airflow, while the other becomes decongested. On average, the congestion pattern switches about every 2 hours, according to a small 2016 study published in the journal PLOS One. \n Reasoning:= This suggests 45 minutes switch time in your statement is wrong."},
                        {"role": "assistant", "content": "Fixed Statement:= Your nose switches back and forth between nostrils. When you sleep, you switch about every 2 hours. This is to prevent a buildup of mucus. It’s called the nasal cycle."},
                        {"role": "user", "content":user_llm }])
            edit_count +=1
            edited_responses.append(response['choices'][0]['message']['content'])
            response_llm=response['choices'][0]['message']['content']
        
    if edit_count==0:
        print('Nothing to edit as the statement seems to be factually correct')
        edited_responses = "Sucessfully fact checked and statement appears to be correct"
        status = True
    else:
        print('Edited Statements Based of the disagreement of facts in documentation found and statement made')
        print('\n')
        print('*************************************************')
        for i in edited_responses:
            print(i)
            print('*************************************************')
        status = False

    return edited_responses, agreement_gates, status

# Function to initialize pinecone index
def pinecone_init(api_key,environment,index_name):
    """
    Create the Pinecone index

    Inputs: api_key (string) - Pinecone API key   
            environment (string) - region where index is to be stored
            index_name (string) - name of index specified by user to be created

    Output: document similarities - Dict of most relevant passages from documents 
    """
    try :
        pinecone.init(api_key=api_key, environment=environment )
        index = pinecone.Index(index_name)
        return index
    except Exception as error:
    # handle the exception
        print("An exception occurred:", error)

def document_preprocessing(text):
    """
       Breaks up texts into contents of managable size (300 words) to uplaod to Pinecone

       Inputs:  text (string) - textual data from document to be preprocessed
       Outputs: split_content (list) - content from texts passed as input split into manageable chunks sizes  
    """
    split_content = []
    try:
        current_content = ""
        current_word_count = 0
        for sentence in re.split("(?<=[.!?]) +", text):
            sentence_word_count = len(sentence.split())
            if current_word_count + sentence_word_count > 300:
                if current_content != "":
                    split_content.append(current_content.strip())
                current_content = sentence
                current_word_count = sentence_word_count
            else:
                current_content += " " + sentence
                current_word_count += sentence_word_count

        if current_content != "":
            split_content.append(current_content.strip())
        return split_content
    except Exception as error:
        print("Failed to preprocess documents: {0}".format(error))
        return False

def upsert_document(openai, split_content, filename, page_num, embedding_model, pineindex, indexname):
    """
       Inserts managable content into pinecone index

       Inputs:  openai (class) - OpenAI API client 
                split_content (lsit) - List of content split into small chunk sizes
                filename (string) - name fo file that is being processed
                page_num (int) - page number of documnet being processed
                embedding_model (string) - name of model used to create embeddings
                indexname (string) - Name of Pinecone index to store embeddings from documents

        output: None
    """
    para=0
    try:
        # Append the split content to the list
        for content in split_content:
            para +=1
            iid= filename[:-4]+'_' +str(page_num)+ '_'+str(para)
            result = openai.Embedding.create(model=embedding_model,input=content)
            embedding=result["data"][0]["embedding"]
            vector = [{'id': iid,
                    'values':embedding,
                    'metadata':{"filename": filename, "word_count": len(content.split()), 'context': content}
                    }]
            pineindex.upsert(vectors=vector, namespace=indexname ) 
            print('Uploaded content to Pinecone index. {0}'.format(vector))
    except Exception as error:
        print("Failed to upload content: {0}".format(error))

def document_upsert_pinecone(openai, embedding_model, pineindex, indexname, filepath):
    """
        Reads the documents and breaks into contents of managable size(300 words) and inserts into pinecone index

        Inputs: openai (class) - OpenAI API client
                embedding_model (string) -
                pineindex (class) - Pinecone API client
                indexname (string) - name of Pinecone index to store embeddings
                filepath (string) - path to where documents are located on EFS

        output: Dictionary with 2 keys: 
                status - confirms if documents were successfully uploaded to pinecone ot not (True/False)
                message - contains string stating if it was successfully uploaded or failed with failure message.
    """
    try:
        print('Processing documents....')
        for filename in os.listdir(filepath):
            if filename.endswith(".pdf"):
                # Open PDF file
                pdf_file = open(os.path.join(filepath, filename), "rb")
                pdf_reader = PyPDF2.PdfReader(pdf_file)

                # Loop through pages and split into documents of 300 tokens
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    content = document_preprocessing(page_text)
                    upsert_document(openai, content, filename, page_num, embedding_model, pineindex, indexname)
                    
            elif filename.endswith(".txt"):    
                # Read text file
                with open(os.path.join(filepath, filename), "r") as f:
                    text = f.read()
                page_num = 0
                content = document_preprocessing(text)
                upsert_document(openai, content, filename, page_num, embedding_model, pineindex, indexname)

            result = {  'status': True,
                        'message': "Successfully processed and uploaded all documents"}
    except Exception as error:
        result = {  'status': False,
                    'message': "Failed: an exception occurred: {0}".format(error)}
    return result


def create_pinecone_index(CreatePineconeIndexRequest):
    """
        Creates a Pinecone index to upload embeddings to
        
        inputs: name (string) - name specified to call index on pinecone
                environment (string) - pincone environment where index is stored

        output: result - Dict with 2 keys: 
                        status  - confirms if index was successfully created or not (True/False)
                        message - contains string stating if it was successfully created or failed with failure message
    """
    index = CreatePineconeIndexRequest['name']
    pine_api_key = os.environ.get('PINECONE_API')
    pine_env = CreatePineconeIndexRequest['environmnet']
    try :
        pinecone.init(api_key=pine_api_key, environment=pine_env)
        pinecone.create_index(index, dimension=1536, metric='cosine', pods=1, replicas=1)
        result = {'status': True,
                  'message': "Successfully created index."}
    except Exception as error:
    # handle the exception
        result = {'status': False,
                  'message': "Failed: an exception occurred: {0}".format(error)}
    return result


def upload_factcheck_documents(UploadFactcheckDocumentRequest):
    """
        Uploads embeddings to Pinecone index.

        inputs: index (string) - name of pinecone index to store embeddings
                environment (string) - pincone environment where index is stored
                fileptah (string) - filepath of where documents to be uploaded are stored

        output: content - Dict with 2 keys: 
                        status - confirms if index was successfully uploaded or not (True/False)
                        message - contains string stating if documents were sucessfully uploaded or failed with failure message.
    """
    pine_api_key = os.environ.get('PINECONE_API')
    openai.api_key = os.environ.get('OPEN_API')
    embedding_model = "text-embedding-ada-002"

    index = UploadFactcheckDocumentRequest['index']
    pine_env = UploadFactcheckDocumentRequest['environment']
    filepath= UploadFactcheckDocumentRequest['filepath']

    pineindex=pinecone_init(pine_api_key,pine_env,index)
    content=document_upsert_pinecone(openai, embedding_model, pineindex, index, filepath)
    return content

def factcheck(factcheckRequest):
    """
        Fact checks a statment given the ground truth docs stored on pinecone index

        inputs: index (string) - name of index where embeddings of ground truth docs are stored on pinecone
                environment (string) - pincone environmnet where index is stored
                statement (string) - the statement to be fact checked  

        output: Dictionary with 3 keys: 
                status - confirms if statement is factually correct or not True/False)
                reasoning - contains reasons why we beleive statement is true or false
                fixed_statement - edited version of statement based on reasons 
    """
    pine_api_key = os.environ.get('PINECONE_API')
    openai.api_key = os.environ.get('OPEN_API')

    index = factcheckRequest['index']
    pine_env = factcheckRequest['environment']
    statement = factcheckRequest['statement']

    pineindex=pinecone_init(pine_api_key,pine_env,index)
    editor_responses, agreement_gates, status =editor(openai, statement, pineindex, index)
   
    result = {  'status': status,
                'reasoning': agreement_gates,
                'fixed_statement': editor_responses}
    return result
import json
import math
import numpy as np
import os
import pandas as pd
import pickle
import re
import requests
import sys
import time
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_text
import torch
from datetime import datetime
import multiprocessing as mp
import copy

from api.term_analysis import chi2_analyzer, similarity_analyzer
from api.utils import pandas_utils
from api.utils.log import getLogger
from enum import Enum
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import current_app
from flask_healthz import healthz
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

embeddingsLogger = getLogger('calculate_embeddings')

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

# to generate test data


def objtofile(data, filename, logger):
    with open('test_data/' + filename + '.obj', 'wb') as outfile:
        pickle.dump(data, outfile)


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

def status_update_worker(logger, log_extras, status_queue, res_queue):
    latest_status_data = None
    while True:
        try:
            status_data = status_queue.get(timeout=10)
            if status_data is not None:
                latest_status_data = status_data
            if latest_status_data is not None:
                logger.info(latest_status_data['json']['statusDescription'], extra=log_extras)
                updated_status_data = copy.deepcopy(latest_status_data)
                updated_status_data['json']['statusDescription'] = updated_status_data['json']['statusDescription'] + ' - Latest status update at ' + datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
                res_queue.put((updated_status_data, None, None))
        except Exception as e:
            if latest_status_data is not None:
                logger.info(latest_status_data['json']['statusDescription'], extra=log_extras)
                updated_status_data = copy.deepcopy(latest_status_data)
                updated_status_data['json']['statusDescription'] = updated_status_data['json']['statusDescription'] + ' - Latest status update at ' + datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
                res_queue.put((updated_status_data, None, None))


def calculate_embeddings_worker(logger, worker_name, req_queue, res_queue, err_queue, embeddingsRequest, method):
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

    filter = embeddingsRequest['filter'] if 'filter' in embeddingsRequest else None
    intents = embeddingsRequest['intents'] if 'intents' in embeddingsRequest else None

    logger.info('Filter is %s', filter, extra=log_extras)

    response_data = {}
    status_data = {}
    if 'boxEndpoint' in embeddingsRequest:
        response_data['boxEndpoint'] = embeddingsRequest['boxEndpoint']
        response_data['header'] = {"content-type": "application/json"}
        status_data['boxEndpoint'] = response_data['boxEndpoint']
        status_data['header'] = response_data['header']

    status_queue = mp.Queue()

    def sendStatus(category, calc_status, step, max_steps, message):
        logger.info(message, extra=log_extras)
        status_data['json'] = {
            "method": "calculate_" + category,
            "clientId": clientId,
            "coachSessionId": coachSessionId,
            "status": calc_status,
            "statusDescription": message,
            "step": step,
            "steps": max_steps
        }
        res_queue.put((status_data, None, None))
        status_queue.put(status_data)

    pstatus = mp.Process(target=status_update_worker, name='status_update_worker', args=(logger, log_extras, status_queue, res_queue))
    pstatus.start()

    if method == "calculate_chi2":
        if len(intents) == 0:
            response_data['json'] = {
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
                response_data['json'], indent=2), extra=log_extras)
            res_queue.put((response_data,))
            return

        try:
            if not 'maxxgrams' in filter:
                filter['maxxgrams'] = 5

            flattenedForChi2 = pandas_utils.flatten_intents_list(intents)
            # objtofile(flattenedForChi2, 'flattenedForChi2', logger)

            sendStatus('chi2', CalcStatus.CHI2_ANALYSIS_RUNNING,
                       1, 4, 'Chi2 Analysis started')
            try:
                chi2, unigram_intent_dict, bigram_intent_dict = chi2_analyzer.get_chi2_analysis(
                    logger, log_extras, worker_name, flattenedForChi2, sendStatus, CalcStatus, num_xgrams=filter['maxxgrams'])
                # objtofile(chi2, 'chi2', logger)
                # objtofile(unigram_intent_dict, 'unigram_intent_dict', logger)
                # objtofile(bigram_intent_dict, 'bigram_intent_dict', logger)
            except Exception as e:
                sendStatus('chi2', CalcStatus.CHI2_ANALYSIS_FAILED,
                           1, 4, 'Chi2 analysis failed - {}'.format(e))
            sendStatus('chi2', CalcStatus.CHI2_ANALYSIS_READY,
                       1, 4, 'Chi2 analysis ready')

            sendStatus('chi2', CalcStatus.CHI2_AMBIGUOS_UNIGRAMS_RUNNING,
                       2, 4, 'Chi2 Ambiguous Unigrams Analysis running')
            try:
                chi2_ambiguous_unigrams = chi2_analyzer.get_confusing_key_terms(
                    unigram_intent_dict)
                # objtofile(chi2_ambiguous_unigrams, 'chi2_ambiguous_unigrams', logger)
            except Exception as e:
                sendStatus('chi2', CalcStatus.CHI2_AMBIGUOS_UNIGRAMS_FAILED, 2,
                           4, 'Chi2 Ambiguous Unigrams Analysis failed - {}'.format(e))
            sendStatus('chi2', CalcStatus.CHI2_AMBIGUOS_UNIGRAMS_READY,
                       2, 4, 'Chi2 Ambiguous Unigrams Analysis ready')

            sendStatus('chi2', CalcStatus.CHI2_AMBIGUOS_BIGRAMS_RUNNING,
                       3, 4, 'Chi2 Ambiguous Bigrams Analysis running')
            try:
                chi2_ambiguous_bigrams = chi2_analyzer.get_confusing_key_terms(
                    bigram_intent_dict)
                # objtofile(chi2_ambiguous_bigrams, 'chi2_ambiguous_bigrams', logger)
            except Exception as e:
                sendStatus('chi2', CalcStatus.CHI2_AMBIGUOS_BIGRAMS_FAILED, 3,
                           4, 'Chi2 Ambiguous Bigrams Analysis failed - {}'.format(e))
            sendStatus('chi2', CalcStatus.CHI2_AMBIGUOS_BIGRAMS_READY,
                       3, 4, 'Chi2 Ambiguous Bigrams Analysis ready')

            sendStatus('chi2', CalcStatus.CHI2_SIMILARITY_ANALYSIS_RUNNING,
                       4, 4, 'Chi2 Similarity Analysis running')
            try:
                chi2_similarity = similarity_analyzer.ambiguous_examples_analysis(
                    logger, log_extras, worker_name, flattenedForChi2, filter['minsimilarity'])
                # objtofile(chi2_similarity, 'chi2_similarity', logger)
            except Exception as e:
                sendStatus('chi2', CalcStatus.CHI2_SIMILARITY_ANALYSIS_FAILED,
                           4, 4, 'Chi2 Similarity Analysis failed - {}'.format(e))
            sendStatus('chi2', CalcStatus.CHI2_SIMILARITY_ANALYSIS_READY,
                       4, 4, 'Chi2 Similarity Analysis ready')
            logger.info('Returning results', extra=log_extras)

            response_data['json'] = {
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
            res_queue.put((response_data,))

        except Exception as e:
            logger.error('Calculating chi2 failed: ' +
                         str(e), extra=log_extras)
            response_data['json'] = {
                "method": "calculate_chi2",
                "status": "failed",
                "coachSessionId": coachSessionId,
                "clientId": clientId,
                "testSetId": testSetId,
                "testSetName": testSetName,
                "err": 'Calculating chi2 failed: ' + str(e)
            }
            res_queue.put((response_data,))

    if method == "calculate_embeddings":
        if len(intents) == 0:
            response_data['json'] = {
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
                response_data['json'], indent=2), extra=log_extras)
            res_queue.put((response_data,))
            return

        sendStatus('embeddings', CalcStatus.EMBEDDINGS_INTENTS_RUNNING, 1, 4, "Loading word embeddings model")

        logger.info('Loading word embeddings model from tfhub ...')
        try:
            generate_embeddings = hub.load(
                'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3')
        except Exception as e:
            err_queue.put(str(e))
        logger.info('Word embeddings model ready.')

        try:
            if not 'maxxgrams' in filter:
                filter['maxxgrams'] = 5

            sendStatus('embeddings', CalcStatus.EMBEDDINGS_INTENTS_RUNNING, 1, 4,
                       "Embeddings calculation for {} intents running".format(len(intents)))
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
                sendStatus('embeddings', CalcStatus.EMBEDDINGS_INTENTS_FAILED, 1, 4,
                           "Embeddings calculation for {} intents failed - {}".format(len(intents), e))
                exit(1)

            sendStatus('embeddings', CalcStatus.EMBEDDINGS_INTENTS_READY, 1, 4,
                       "Embeddings calculation for {} intents ready".format(len(intents)))

            sendStatus('embeddings', CalcStatus.EMBEDDINGS_PCA_RUNNING,
                       2, 4, 'Principal Component Analysis running')
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
                sendStatus('embeddings', CalcStatus.EMBEDDINGS_PCA_FAILED,
                           2, 4, 'Principal Component Analysis failed - {}'.format(e))
                exit(1)
            sendStatus('embeddings', CalcStatus.EMBEDDINGS_PCA_READY,
                       2, 4, 'Principal Component Analysis ready')

            sendStatus('embeddings', CalcStatus.EMBEDDINGS_PREPARE_COSINE_SIMILARITY_RUNNING,
                       3, 4, f'Preparation for Cosine Similarity Analysis for {len(flattenedForCosine)} examples running')
            try:
                logger.info('Preparing cosine similarity for %s examples', len(flattenedForCosine), extra=log_extras)

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
                sendStatus('embeddings', CalcStatus.EMBEDDINGS_PREPARE_COSINE_SIMILARITY_FAILED,
                           3, 4, 'Preparation for Cosine Similarity Analysis failed - {}'.format(e))
                exit(1)

            #sendStatus('embeddings', CalcStatus.EMBEDDINGS_PREPARE_COSINE_SIMILARITY_READY,
            #           3, 4, 'Preparation for Cosine Similarity Analysis ready')

            #sendStatus('embeddings', CalcStatus.EMBEDDINGS_COSINE_SIMILARITY_RUNNING,
            #           4, 4, f'Cosine Similarity Analysis running for {len(flattenedForCosine)} examples ')
            logger.info('Running cosine similarity for %s examples', len(flattenedForCosine), extra=log_extras)

            # data = Parallel(n_jobs=-1)(delayed(cosine_similarity_worker)(w[0], w[1], w[2], w[3], w[4], w[5]) for w in workers)
            executer = ThreadPoolExecutor(max_workers=os.environ.get('COACH_THREADS_EMBEDDINGS_COSINE_SIMILARITY', 3))
            data = list()
            for result in executer.map(cosine_similarity_worker, tuple(workers)):
                data.append(result)
                progress = len(data)
                if progress % 5000 == 0:
                    sendStatus('embeddings', CalcStatus.EMBEDDINGS_COSINE_SIMILARITY_RUNNING,
                        4, 4, f'Cosine Similarity Analysis calculations {int(progress * 100/len(workers))}% ({progress}/{len(workers)})')
            #data = list(executer.map(cosine_similarity_worker, tuple(workers)))

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
            logger.debug(json.dumps(similarity, indent=2), extra=log_extras)

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
            logger.debug(json.dumps(separation, indent=2), extra=log_extras)
            sendStatus('embeddings', CalcStatus.EMBEDDINGS_COSINE_SIMILARITY_READY,
                       4, 4, 'Cosine Similarity Analysis ready')

            logger.info('Returning results', extra=log_extras)

            response_data['json'] = {
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
            logger.debug(json.dumps(response_data, indent=2), extra=log_extras)
            res_queue.put((response_data,))
        except Exception as e:
            logger.error('Calculating embeddings failed: ' +
                         str(e), extra=log_extras)
            response_data['json'] = {
                "method": "calculate_embeddings",
                "status": "failed",
                "coachSessionId": coachSessionId,
                "err": 'Calculating embeddings failed: ' + str(e)
            }
            logger.debug(json.dumps(response_data, indent=2))
            res_queue.put((response_data,))

    pstatus.terminate()
    pstatus.join()

def ping():
    return 'Botium Coach Worker. Tensorflow Version: {tfVersion} PyTorch Version: {ptVersion}, Cuda: {ptCuda}'.format(
        tfVersion=tf.__version__, ptVersion=torch.__version__, ptCuda=str(torch.cuda.is_available()))

def calculate_embeddings(embeddingsRequest):
    coachSessionId = embeddingsRequest['coachSessionId'] if 'coachSessionId' in embeddingsRequest else None
    clientId = embeddingsRequest['clientId'] if 'clientId' in embeddingsRequest else None
    testSetId = embeddingsRequest['testSetId'] if 'testSetId' in embeddingsRequest else None
    testSetName = embeddingsRequest['testSetName'] if 'testSetName' in embeddingsRequest else None

    boxEndpoint = embeddingsRequest['boxEndpoint'] if 'boxEndpoint' in embeddingsRequest else None
    if 'COACH_DEV_BOX_ENDPOINT' in os.environ:
        boxEndpoint = os.environ.get('COACH_DEV_BOX_ENDPOINT')

    try:
        embeddingsLogger.info('Checking callback url availability (' + boxEndpoint + ') ...')
        response_data = {
            "method": "ping"
        }
        res = requests.post(boxEndpoint, json=response_data)
        if res.status_code != 200 and res.status_code != 400:
            raise Exception(
                'Ping check for callback url failed: Status Code ' + str(res.status_code))
    except Exception as e:
        embeddingsLogger.info('Error: Checking callback url availability: ' + str(e))
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

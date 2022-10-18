import os
import json
from .utils.log import getLogger 
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_text
import numpy as np
import pandas as pd
import math
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Parallel, delayed
from api.term_analysis import chi2_analyzer, similarity_analyzer
from api.utils import pandas_utils
from flask import current_app
import torch
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import redis

from functools import singledispatch


@singledispatch
def to_serializable(val):
    """Used by default."""
    return str(val)


@to_serializable.register(np.float32)
def ts_float32(val):
    """Used if *val* is an instance of numpy.float32."""
    return np.float64(val)

maxUtterancesForEmbeddings = -1
if 'COACH_MAX_UTTERANCES_FOR_EMBEDDINGS' in os.environ:
  maxUtterancesForEmbeddings = int(os.environ['COACH_MAX_UTTERANCES_FOR_EMBEDDINGS'])
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

def calculate_embeddings_worker(req_queue, processId):
    red = redis.Redis(host='localhost', port=6379, db=0) 
    worker_name = 'Worker ' + str(processId)
    logger = getLogger(worker_name)
    logger.info('Initialize worker ...')
    logger.info('Loading word embeddings model from tfhub ...')
    generate_embeddings = hub.load('https://tfhub.dev/google/universal-sentence-encoder-multilingual/3')
    logger.info('Word embeddings model ready.')
    logger.info('Worker started!')
    calc_count = 0
    while calc_count < maxCalcCount:
        embeddingsRequest, method = req_queue.get()
        logger.debug(json.dumps(embeddingsRequest, indent=2, default=to_serializable))
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
        stat = {
            "embeddings": [],
            "chi2": []
        }
        def sendStatus(method, status):
            stat[method].append(status)
            logger.info('Sending %s to redis', status, extra=log_extras)
            red.set('coachworker_status_' + method + '_' + coachSessionId, json.dumps({
                "method": "calculate_" + method,
                "clientId": clientId,
                "coachSessionId": coachSessionId,
                "status": stat[method]
            }))
        # for testing purposes on local environment
        if 'COACH_DEV_BOX_ENDPOINT' in os.environ:
            boxEndpoint = os.environ.get('COACH_DEV_BOX_ENDPOINT')
        if method == "retryRequest":
            seconds = int(os.environ.get('COACH_RETRY_REQUEST_DELAY', 10))
            max_retries = int(os.environ.get('COACH_RETRY_REQUEST_RETRIES', 12))
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
                        res = requests.post(boxEndpoint, headers = embeddingsRequest["header"], json = embeddingsRequest["json"])
                    else:
                        res = requests.post(boxEndpoint, headers = embeddingsRequest["header"], data = embeddingsRequest["data"])
                else:
                    if 'json' in embeddingsRequest.keys():
                        res = requests.post(boxEndpoint, json = embeddingsRequest["json"])
                    else:
                        res = requests.post(boxEndpoint, data = embeddingsRequest["data"])
                if res.status_code != 200:
                    raise Exception('Wrong status code ' + str(res.status_code))
                logger.info(str(res), extra=log_extras)
                logger.info('retry request for %s ( %s of %s ) to %s successfully sent',
                    embeddingsRequest["retry_method"],
                    retry,
                    max_retries,
                    boxEndpoint, 
                    extra=log_extras)
            except Exception as e:
                logger.info('%s: %s', worker_name, e)
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
                    logger.debug(json.dumps(response_data, indent=2), extra=log_extras)
                    if boxEndpoint is not None:
                        try:
                            res = requests.post(boxEndpoint, json = response_data)
                            if res.status_code != 200:
                                raise Exception('Wrong status code ' + str(res.status_code))
                            logger.info(str(res), extra=log_extras)
                        except Exception as e:
                            logger.error('Sending chi2 failed: ' + str(e), extra=log_extras)
                            req_queue.put(({
                                "boxEndpoint": boxEndpoint,
                                "json": response_data,
                                "retry_method": "calculate_chi2"
                            }, "retryRequest"))
                    else:
                        red.set('coachworker_res_chi2_' + coachSessionId, json.dumps(response_data))
                        red.delete('coachworker_req_chi2_' + coachSessionId)
                    continue

                if not 'maxxgrams' in filter:
                  filter['maxxgrams'] = 5

                flattenedForChi2 = pandas_utils.flatten_intents_list(intents)

                logger.info('Running chi2 analysis', extra=log_extras)
                sendStatus('chi2', 'Step 1 / 4: Running chi2 analysis')
                chi2, unigram_intent_dict, bigram_intent_dict = chi2_analyzer.get_chi2_analysis(logger, worker_name, flattenedForChi2, num_xgrams=filter['maxxgrams'])

                logger.info('Running chi2 ambiguous unigrams analysis', extra=log_extras)
                sendStatus('chi2', 'Step 2 / 4: Running chi2 ambiguous unigrams analysis')
                chi2_ambiguous_unigrams = chi2_analyzer.get_confusing_key_terms(unigram_intent_dict)

                logger.info('Running chi2 ambiguous bigrams analysis', extra=log_extras)
                sendStatus('chi2', 'Step 3 / 4: Running chi2 ambiguous bigrams analysis')
                chi2_ambiguous_bigrams = chi2_analyzer.get_confusing_key_terms(bigram_intent_dict)

                logger.info('Running chi2 similarity analysis', extra=log_extras)
                sendStatus('chi2', 'Step 4 / 4: Running chi2 similarity analysis')
                chi2_similarity = similarity_analyzer.ambiguous_examples_analysis(logger, worker_name, flattenedForChi2, filter['minsimilarity'])
                logger.info('Returning results', extra=log_extras)

                logger.info('Sending results to %s', boxEndpoint, extra=log_extras)
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
                        res = requests.post(boxEndpoint, headers = header, data = data)
                        if res.status_code != 200:
                            raise Exception('Wrong status code ' + str(res.status_code))
                        logger.info(str(res), extra=log_extras)
                    except Exception as e:
                        logger.error('Sending chi2 failed: ' + str(e), extra=log_extras)
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
                logger.error('Calculating chi2 failed: ' + str(e), extra=log_extras)
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
                    res = requests.post(boxEndpoint, json = response_data)
                    if res.status_code != 200:
                        raise Exception('Wrong status code ' + str(res.status_code))
                    logger.info('%s: ' + str(res), worker_name)
                except Exception as e:
                    logger.error('Sending chi2 failed: ' + str(e), extra=log_extras)
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
                    logger.debug(json.dumps(response_data, indent=2), extra=log_extras)
                    if boxEndpoint is not None:
                        try:
                            res = requests.post(boxEndpoint, json = response_data)
                            if res.status_code != 200:
                                raise Exception('Wrong status code ' + str(res.status_code))
                            logger.info(str(res), extra=log_extras)
                        except Exception as e:
                            logger.error('Sending embeddings failed: ' + str(e), extra=log_extras)
                            req_queue.put(({
                                "boxEndpoint": boxEndpoint,
                                "json": response_data,
                                "retry_method": "calculate_embeddings"
                            }, "retryRequest"))
                    else:
                        red.set('coachworker_res_embeddings_' + coachSessionId, json.dumps(response_data))
                        red.delete('coachworker_req_embeddings_' + coachSessionId)
                    continue

                if not 'maxxgrams' in filter:
                  filter['maxxgrams'] = 5

                

                logger.info('Calculating embeddings for %s intents', len(intents), extra=log_extras)
                sendStatus('embeddings', 'Step 1 / 4: Calculating embeddings for intents')
                for intent in intents:
                  logger.info('Calculating embeddings for intent "%s" with %s: examples', intent['name'], len(intent['examples']), extra=log_extras)

                training_phrases_with_embeddings = defaultdict(list)
                for intent in intents:
                  if len(intent['examples']) > 0:
                    computed_embeddings = generate_embeddings(intent['examples'])
                    training_phrases_with_embeddings[intent['name']] = dict(zip(intent['examples'], computed_embeddings))

                for intent_name, _ in training_phrases_with_embeddings.items():
                  training_phrase, embeddings = next(iter(training_phrases_with_embeddings[intent_name].items()))
                  logger.info('Calculated embeddings for intent {}, example: {{\'{}\':{}}}'.format(intent_name, training_phrase, embeddings[:5]), extra=log_extras)

                embedding_vectors = []

                for intent, training_phrases_and_embeddings in training_phrases_with_embeddings.items():
                  for training_phrase, embeddings in training_phrases_and_embeddings.items():
                    embedding_vectors.append(embeddings)

                embedding_vectors = np.asarray(embedding_vectors)

                sendStatus('embeddings', 'Step 2 / 4: Starting principal component analysis')
                logger.info('Starting principal component analysis for %s examples', len(embedding_vectors), extra=log_extras)

                pca = PCA(n_components=2)
                pca.fit(embedding_vectors)

                embeddings_coords = []

                for color, intent in enumerate(training_phrases_with_embeddings.keys()):
                  phrases = list(training_phrases_with_embeddings[intent].keys())
                  embeddings = list(training_phrases_with_embeddings[intent].values())
                  points = pca.transform(embeddings)
                  embeddings_coords.append({
                    'name': intent,
                    'examples': [ { 'phrase': phrase, 'x': np.float(point[0]), 'y': np.float(point[1]) } for phrase, point in zip(phrases, points)]
                  })

                logger.debug(json.dumps(embeddings_coords, indent=2))
                logger.info('Ready with principal component analysis for %s examples', len(embedding_vectors), extra=log_extras)

                flattenedForCosine = []

                for intent in training_phrases_with_embeddings:
                  phrases = list(training_phrases_with_embeddings[intent].keys())
                  if maxUtterancesForEmbeddings > 0:
                    utterancesForIntent = math.ceil(len(phrases) * maxUtterancesForEmbeddings / len(embedding_vectors))
                    if utterancesForIntent < len(phrases):
                      logger.info('Randomly selecting %s: examples for intent %s: for cosine similarity', utterancesForIntent, intent, extra=log_extras)
                      phrases = np.random.choice(phrases, utterancesForIntent, replace=False)
                  for phrase in phrases:
                    flattenedForCosine.append((intent, phrase, training_phrases_with_embeddings[intent][phrase]))

                sendStatus('embeddings', 'Step 3 / 4: Preparing cosine similarity')
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

                    workers.append((intent_1, phrase_1, embedd_1, intent_2, phrase_2, embedd_2))

                sendStatus('embeddings', 'Step 4 / 4: Running cosine similarity')
                logger.info('Running cosine similarity for %s examples', len(flattenedForCosine), extra=log_extras)

                # data = Parallel(n_jobs=-1)(delayed(cosine_similarity_worker)(w[0], w[1], w[2], w[3], w[4], w[5]) for w in workers)
                executer = ThreadPoolExecutor(max_workers = os.environ.get('COACH_THREADS_EMBEDDINGS_COSINE_SIMILARITY', 3))
                data = list(executer.map(cosine_similarity_worker, tuple(workers)))

                logger.info('Ready with cosine similarity for %s pairs, preparing results', len(data), extra=log_extras)

                similarity_df = pd.DataFrame(data, columns=['name1', 'example1', 'name2', 'example2', 'similarity'])
                similarity_different_intent = similarity_df['name1'] != similarity_df['name2']
                similarity_same_intent = similarity_df['name1'] == similarity_df['name2']

                similarity_different_intent_filtered = (similarity_df['name1'] != similarity_df['name2']) & (similarity_df['similarity'] > filter['minsimilarity'])
                similarity_df_sorted = similarity_df[similarity_different_intent_filtered].sort_values('similarity', ascending=False)
                similarity = [ { 'name1': name1, 'example1': example1, 'name2': name2, 'example2': example2, 'similarity': similarity } for name1, example1, name2, example2, similarity in zip(similarity_df_sorted['name1'], similarity_df_sorted['example1'], similarity_df_sorted['name2'], similarity_df_sorted['example2'], similarity_df_sorted['similarity'])]
                logger.debug(json.dumps(similarity, indent=2), extra=log_extras)

                cohesion_df_sorted = pd.DataFrame(similarity_df[similarity_same_intent].groupby('name1', as_index=False)['similarity'].mean()).sort_values('similarity', ascending=False)
                cohesion_df_sorted.columns = ['name', 'cohesion']
                cohesion = [ { 'name': name, 'cohesion': cohesion } for name, cohesion in zip(cohesion_df_sorted['name'], cohesion_df_sorted['cohesion'])]
                logger.debug(json.dumps(cohesion, indent=2), extra=log_extras)

                separation_df_sorted = pd.DataFrame(similarity_df[similarity_different_intent].groupby(['name1', 'name2'], as_index=False)['similarity'].mean()).sort_values('similarity', ascending=True)
                separation_df_sorted['separation'] = 1 - separation_df_sorted['similarity']
                separation = [ { 'name1': name1, 'name2': name2, 'separation': separation } for name1, name2, separation in zip(separation_df_sorted['name1'], separation_df_sorted['name2'], separation_df_sorted['separation'])]
                logger.debug(json.dumps(separation, indent=2), extra=log_extras)

                logger.info('Returning results', extra=log_extras)

                logger.info('Sending results to %s', boxEndpoint, extra=log_extras)
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
                logger.debug(json.dumps(response_data, indent=2), extra=log_extras)
                if boxEndpoint is not None:
                    try:
                        res = requests.post(boxEndpoint, json = response_data)
                        if res.status_code != 200:
                            raise Exception('Wrong status code ' + str(res.status_code))
                        logger.info(str(res), extra=log_extras)
                    except Exception as e:
                        logger.error('Sending embeddings failed: ' + str(e), extra=log_extras)
                        req_queue.put(({
                            "boxEndpoint": boxEndpoint,
                            "json": response_data,
                            "retry_method": "calculate_embeddings"
                        }, "retryRequest"))
                else:
                    red.set('coachworker_res_embeddings_' + coachSessionId, json.dumps(response_data))
                    red.delete('coachworker_req_embeddings_' + coachSessionId)
                calc_count += 1
            except Exception as e:
                logger.error('Calculating embeddings failed: ' + str(e), extra=log_extras)
                response_data = {
                    "method": "calculate_embeddings",
                    "status": "failed",
                    "coachSessionId": coachSessionId,
                    "err": 'Calculating embeddings failed: ' + str(e)
                }
                logger.debug(json.dumps(response_data, indent=2))
                try:
                    res = requests.post(boxEndpoint, json = response_data)
                    if res.status_code != 200:
                        raise Exception('Wrong status code ' + str(res.status_code))
                    logger.info(str(res), extra=log_extras)
                except Exception as e:
                    logger.error('Sending embeddings failed: ' + str(e), extra=log_extras)
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
      res = requests.post(boxEndpoint, json = response_data)
      if res.status_code != 200 and res.status_code != 400:
          raise Exception('Ping check for callback url failed: Status Code ' + str(res.status_code))
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

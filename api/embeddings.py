import os
import json
import logging
import tensorflow as tf
import tensorflow_hub as hub
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

def calculate_embeddings_worker(req_queue, processId, log_format, log_level, log_datefmt):
    worker_name = 'Worker ' + str(processId)
    logging.basicConfig(format=log_format, level=log_level, datefmt=log_datefmt)
    logger = logging.getLogger(worker_name)
    logger.info('%s: Initialize worker ...', worker_name)
    logger.info('%s: Loading word embeddings model from tfhub ...', worker_name)
    generate_embeddings = hub.load('https://tfhub.dev/google/universal-sentence-encoder-multilingual/3')
    logger.info('%s: Word embeddings model ready.', worker_name)
    logger.info('%s: Worker started!', worker_name)
    calc_count = 0
    while calc_count < maxCalcCount:
        embeddingsRequest, method = req_queue.get()
        logger.debug(json.dumps(embeddingsRequest, indent=2, default=to_serializable))
        coachSessionId = embeddingsRequest['coachSessionId'] if 'coachSessionId' in embeddingsRequest else None
        boxEndpoint = embeddingsRequest['boxEndpoint']
        filter = embeddingsRequest['filter'] if 'filter' in embeddingsRequest else None
        intents = embeddingsRequest['intents'] if 'intents' in embeddingsRequest else None
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
            logger.info('%s: next retry request for %s ( %s of %s ) in %s seconds',
                worker_name,
                embeddingsRequest["retry_method"],
                retry,
                max_retries,
                seconds
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
                logger.info('%s: ' + str(res), worker_name)
                logger.info('%s: retry request for %s ( %s of %s ) to %s successfully sent',
                    worker_name,
                    embeddingsRequest["retry_method"],
                    retry,
                    max_retries,
                    boxEndpoint
                )
            except Exception as e:
                logger.info('%s: %s', worker_name, e)
                if retry <= max_retries:
                    logger.info('%s: retry request for %s ( %s of %s ) to %s failed, trying again',
                        worker_name,
                        embeddingsRequest["retry_method"],
                        retry,
                        max_retries,
                        boxEndpoint
                    )
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
                    logger.info('%s: retry request for %s ( %s of %s ) to %s failed, no tries anymore',
                        worker_name,
                        embeddingsRequest["retry_method"],
                        retry,
                        max_retries,
                        boxEndpoint
                    )
        if method == "calculate_chi2":
            try:
                if len(intents) == 0:
                    response_data = {
                        "method": "calculate_chi2",
                        "status": "finished",
                        "coachSessionId": coachSessionId,
                        "output": {
                          'chi2': [],
                          'chi2_ambiguous_unigrams': [],
                          'chi2_ambiguous_bigrams': [],
                          'chi2_similarity': []
                        }
                    }
                    logger.debug('%s: ' + json.dumps(response_data, indent=2), worker_name)
                    try:
                        res = requests.post(boxEndpoint, json = response_data)
                        if res.status_code != 200:
                            raise Exception('Wrong status code ' + str(res.status_code))
                        logger.info('%s: ' + str(res), worker_name)
                    except Exception as e:
                        logger.error('%s: Sending chi2 failed: ' + str(e), worker_name)
                        req_queue.put(({
                            "boxEndpoint": boxEndpoint,
                            "json": response_data,
                            "retry_method": "calculate_chi2"
                        }, "retryRequest"))
                    continue

                if not 'maxxgrams' in filter:
                  filter['maxxgrams'] = 5

                flattenedForChi2 = pandas_utils.flatten_intents_list(intents)
                logger.info('%s: Running chi2 analysis', worker_name)
                chi2, unigram_intent_dict, bigram_intent_dict = chi2_analyzer.get_chi2_analysis(logger, worker_name, flattenedForChi2, num_xgrams=filter['maxxgrams'])

                logger.info('%s: Running chi2 ambiguous unigrams analysis', worker_name)
                chi2_ambiguous_unigrams = chi2_analyzer.get_confusing_key_terms(unigram_intent_dict)

                logger.info('%s: Running chi2 ambiguous bigrams analysis', worker_name)
                chi2_ambiguous_bigrams = chi2_analyzer.get_confusing_key_terms(bigram_intent_dict)

                logger.info('%s: Running chi2 similarity analysis', worker_name)
                chi2_similarity = similarity_analyzer.ambiguous_examples_analysis(logger, worker_name, flattenedForChi2, filter['minsimilarity'])
                logger.info('%s: Returning results', worker_name)

                logger.info('%s: Sending results to %s', worker_name, boxEndpoint)
                response_data = {
                    "method": "calculate_chi2",
                    "status": "finished",
                    "coachSessionId": coachSessionId,
                    "output": {
                      'chi2': chi2,
                      'chi2_ambiguous_unigrams': chi2_ambiguous_unigrams,
                      'chi2_ambiguous_bigrams': chi2_ambiguous_bigrams,
                      'chi2_similarity': chi2_similarity
                    }
                }
                header = {"content-type": "application/json"}
                data = json.dumps(response_data, default=to_serializable)
                try:
                    res = requests.post(boxEndpoint, headers = header, data = data)
                    if res.status_code != 200:
                        raise Exception('Wrong status code ' + str(res.status_code))
                    logger.info('%s: ' + str(res), worker_name)
                except Exception as e:
                    logger.error('%s: Sending chi2 failed: ' + str(e), worker_name)
                    req_queue.put(({
                        "boxEndpoint": boxEndpoint,
                        "header": header,
                        "data": data,
                        "retry_method": "calculate_chi2"
                    }, "retryRequest"))
                calc_count += 1
            except Exception as e:
                logger.error('%s: Calculating chi2 failed: ' + str(e), worker_name)
                response_data = {
                    "method": "calculate_chi2",
                    "status": "failed",
                    "coachSessionId": coachSessionId,
                    "err": 'Calculating chi2 failed: ' + str(e)
                }
                logger.debug(json.dumps(response_data, indent=2))
                try:
                    res = requests.post(boxEndpoint, json = response_data)
                    if res.status_code != 200:
                        raise Exception('Wrong status code ' + str(res.status_code))
                    logger.info('%s: ' + str(res), worker_name)
                except Exception as e:
                    logger.error('%s: Sending chi2 failed: ' + str(e), worker_name)
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
                        "output": {
                          'embeddings': embeddings_coords,
                          'similarity': similarity,
                          'cohesion': cohesion,
                          'separation': separation,
                        }
                    }
                    logger.debug('%s: ' + json.dumps(response_data, indent=2), worker_name)
                    try:
                        res = requests.post(boxEndpoint, json = response_data)
                        if res.status_code != 200:
                            raise Exception('Wrong status code ' + str(res.status_code))
                        logger.info('%s: ' + str(res), worker_name)
                    except Exception as e:
                        logger.error('%s: Sending embeddings failed: ' + str(e), worker_name)
                        req_queue.put(({
                            "boxEndpoint": boxEndpoint,
                            "json": response_data,
                            "retry_method": "calculate_embeddings"
                        }, "retryRequest"))
                    continue

                if not 'maxxgrams' in filter:
                  filter['maxxgrams'] = 5

                logger.info('%s: Calculating embeddings for %s intents', worker_name, len(intents))
                for intent in intents:
                  logger.info('%s: Calculating embeddings for intent "%s" with %s: examples', worker_name, intent['name'], len(intent['examples']))

                training_phrases_with_embeddings = defaultdict(list)
                for intent in intents:
                  if len(intent['examples']) > 0:
                    computed_embeddings = generate_embeddings(intent['examples'])
                    training_phrases_with_embeddings[intent['name']] = dict(zip(intent['examples'], computed_embeddings))

                for intent_name, _ in training_phrases_with_embeddings.items():
                  training_phrase, embeddings = next(iter(training_phrases_with_embeddings[intent_name].items()))
                  logger.info('{}: Calculated embeddings for intent {}, example: {{\'{}\':{}}}'.format(worker_name, intent_name, training_phrase, embeddings[:5]))

                embedding_vectors = []

                for intent, training_phrases_and_embeddings in training_phrases_with_embeddings.items():
                  for training_phrase, embeddings in training_phrases_and_embeddings.items():
                    embedding_vectors.append(embeddings)

                embedding_vectors = np.asarray(embedding_vectors)

                logger.info('%s: Starting principal component analysis for %s examples', worker_name, len(embedding_vectors))

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
                logger.info('%s: Ready with principal component analysis for %s examples', worker_name, len(embedding_vectors))

                flattenedForCosine = []

                for intent in training_phrases_with_embeddings:
                  phrases = list(training_phrases_with_embeddings[intent].keys())
                  if maxUtterancesForEmbeddings > 0:
                    utterancesForIntent = math.ceil(len(phrases) * maxUtterancesForEmbeddings / len(embedding_vectors))
                    if utterancesForIntent < len(phrases):
                      logger.info('%s: Randomly selecting %s: examples for intent %s: for cosine similarity', worker_name, utterancesForIntent, intent)
                      phrases = np.random.choice(phrases, utterancesForIntent, replace=False)
                  for phrase in phrases:
                    flattenedForCosine.append((intent, phrase, training_phrases_with_embeddings[intent][phrase]))

                logger.info('%s: Preparing cosine similarity for %s examples', worker_name, len(flattenedForCosine))

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

                logger.info('%s: Running cosine similarity for %s examples', worker_name, len(flattenedForCosine))

                # data = Parallel(n_jobs=-1)(delayed(cosine_similarity_worker)(w[0], w[1], w[2], w[3], w[4], w[5]) for w in workers)
                executer = ThreadPoolExecutor(max_workers = os.environ.get('COACH_THREADS_EMBEDDINGS_COSINE_SIMILARITY', 3))
                data = list(executer.map(cosine_similarity_worker, tuple(workers)))

                logger.info('%s: Ready with cosine similarity for %s pairs, preparing results', worker_name, len(data))

                similarity_df = pd.DataFrame(data, columns=['name1', 'example1', 'name2', 'example2', 'similarity'])
                similarity_different_intent = similarity_df['name1'] != similarity_df['name2']
                similarity_same_intent = similarity_df['name1'] == similarity_df['name2']

                similarity_different_intent_filtered = (similarity_df['name1'] != similarity_df['name2']) & (similarity_df['similarity'] > filter['minsimilarity'])
                similarity_df_sorted = similarity_df[similarity_different_intent_filtered].sort_values('similarity', ascending=False)
                similarity = [ { 'name1': name1, 'example1': example1, 'name2': name2, 'example2': example2, 'similarity': similarity } for name1, example1, name2, example2, similarity in zip(similarity_df_sorted['name1'], similarity_df_sorted['example1'], similarity_df_sorted['name2'], similarity_df_sorted['example2'], similarity_df_sorted['similarity'])]
                logger.debug('%s: ' + json.dumps(similarity, indent=2), worker_name)

                cohesion_df_sorted = pd.DataFrame(similarity_df[similarity_same_intent].groupby('name1', as_index=False)['similarity'].mean()).sort_values('similarity', ascending=False)
                cohesion_df_sorted.columns = ['name', 'cohesion']
                cohesion = [ { 'name': name, 'cohesion': cohesion } for name, cohesion in zip(cohesion_df_sorted['name'], cohesion_df_sorted['cohesion'])]
                logger.debug('%s: ' + json.dumps(cohesion, indent=2), worker_name)

                separation_df_sorted = pd.DataFrame(similarity_df[similarity_different_intent].groupby(['name1', 'name2'], as_index=False)['similarity'].mean()).sort_values('similarity', ascending=True)
                separation_df_sorted['separation'] = 1 - separation_df_sorted['similarity']
                separation = [ { 'name1': name1, 'name2': name2, 'separation': separation } for name1, name2, separation in zip(separation_df_sorted['name1'], separation_df_sorted['name2'], separation_df_sorted['separation'])]
                logger.debug('%s: ' + json.dumps(separation, indent=2), worker_name)

                logger.info('%s: Returning results', worker_name)

                logger.info('%s: Sending results to %s', worker_name, boxEndpoint)
                response_data = {
                    "method": "calculate_embeddings",
                    "status": "finished",
                    "coachSessionId": coachSessionId,
                    "output": {
                      'embeddings': embeddings_coords,
                      'similarity': similarity,
                      'cohesion': cohesion,
                      'separation': separation
                    }
                }
                logger.debug('%s: ' + json.dumps(response_data, indent=2), worker_name)
                try:
                    res = requests.post(boxEndpoint, json = response_data)
                    if res.status_code != 200:
                        raise Exception('Wrong status code ' + str(res.status_code))
                    logger.info('%s: ' + str(res), worker_name)
                except Exception as e:
                    logger.error('%s: Sending embeddings failed: ' + str(e), worker_name)
                    req_queue.put(({
                        "boxEndpoint": boxEndpoint,
                        "json": response_data,
                        "retry_method": "calculate_embeddings"
                    }, "retryRequest"))
                calc_count += 1
            except Exception as e:
                logger.error('%s: Calculating embeddings failed: ' + str(e), worker_name)
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
                    logger.info('%s: ' + str(res), worker_name)
                except Exception as e:
                    logger.error('%s: Sending embeddings failed: ' + str(e), worker_name)
                    req_queue.put(({
                        "boxEndpoint": boxEndpoint,
                        "json": response_data,
                        "retry_method": "calculate_embeddings"
                    }, "retryRequest"))

def ping():
  return 'Botium Coach Worker. Tensorflow Version: {tfVersion} PyTorch Version: {ptVersion}, Cuda: {ptCuda}'.format(
    tfVersion=tf.__version__, ptVersion=torch.__version__, ptCuda=str(torch.cuda.is_available()))

def calculate_embeddings(embeddingsRequest):

  coachSessionId = embeddingsRequest['coachSessionId']
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
    'boxEndpoint': boxEndpoint,
    'workerEndpoint': os.environ.get('COACH_HOSTNAME', '')
  }

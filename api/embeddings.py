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

maxUtterancesForEmbeddings = -1
if 'COACH_MAX_UTTERANCES_FOR_EMBEDDINGS' in os.environ:
  maxUtterancesForEmbeddings = int(os.environ['COACH_MAX_UTTERANCES_FOR_EMBEDDINGS'])
maxCalcCount = 100
if 'COACH_MAX_CALCULATIONS_PER_WORKER' in os.environ:
  maxCalcCount = int(os.environ['COACH_MAX_CALCULATIONS_PER_WORKER'])

def cosine_similarity_worker(intent_1, phrase_1, embedd_1, intent_2, phrase_2, embedd_2):
  similarity = cosine_similarity([embedd_1], [embedd_2])[0][0]
  return [intent_1, phrase_1, intent_2, phrase_2, similarity]

def calculate_embeddings_worker(req_queue, processId, log_format, log_level, log_datefmt):
    logging.basicConfig(format='%(name): ' + log_format, level=log_level, datefmt=log_datefmt)
    logger = logging.getLogger('Worker ' + str(processId))
    logger.info(' Initialize worker ...')
    logger.info('Loading word embeddings model from tfhub ...')
    generate_embeddings = hub.load('https://tfhub.dev/google/universal-sentence-encoder-multilingual/3')
    logger.info('Word embeddings model ready.')
    logger.info('Worker started!')
    calc_count = 0
    while calc_count < maxCalcCount:
        embeddingsRequest = req_queue.get()
        try:
            logger.debug(json.dumps(embeddingsRequest, indent=2))
            coachSessionId = embeddingsRequest['coachSessionId']
            boxEndpoint = embeddingsRequest['boxEndpoint']
            filter = embeddingsRequest['filter']
            intents = embeddingsRequest['intents']
            # for testing purposes on local environment
            if 'localhost' in boxEndpoint or '127.0.0.1' in boxEndpoint:
                boxEndpoint = boxEndpoint.replace('3000', '4000')
            if len(intents) == 0:
              return { 'embeddings': [], 'similarity': [], 'cohesion': [], 'separation': [] }

            if not 'maxxgrams' in filter:
              filter['maxxgrams'] = 5

            logger.info('Calculating embeddings for %s intents', len(intents))
            for intent in intents:
              logger.info('Calculating embeddings for intent "%s" with %s examples', intent['name'], len(intent['examples']))

            training_phrases_with_embeddings = defaultdict(list)
            for intent in intents:
              if len(intent['examples']) > 0:
                computed_embeddings = generate_embeddings(intent['examples'])
                training_phrases_with_embeddings[intent['name']] = dict(zip(intent['examples'], computed_embeddings))

            for intent_name, _ in training_phrases_with_embeddings.items():
              training_phrase, embeddings = next(iter(training_phrases_with_embeddings[intent_name].items()))
              logger.info('Calculated embeddings for intent {}, example: {{\'{}\':{}}}'.format(intent_name, training_phrase, embeddings[:5]))

            embedding_vectors = []

            for intent, training_phrases_and_embeddings in training_phrases_with_embeddings.items():
              for training_phrase, embeddings in training_phrases_and_embeddings.items():
                embedding_vectors.append(embeddings)

            embedding_vectors = np.asarray(embedding_vectors)

            logger.info('Starting principal component analysis for %s examples', len(embedding_vectors))

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
            logger.info('Ready with principal component analysis for %s examples', len(embedding_vectors))

            flattenedForCosine = []

            for intent in training_phrases_with_embeddings:
              phrases = list(training_phrases_with_embeddings[intent].keys())
              if maxUtterancesForEmbeddings > 0:
                utterancesForIntent = math.ceil(len(phrases) * maxUtterancesForEmbeddings / len(embedding_vectors))
                if utterancesForIntent < len(phrases):
                  logger.info('Randomly selecting %s examples for intent %s for cosine similarity', utterancesForIntent, intent)
                  phrases = np.random.choice(phrases, utterancesForIntent, replace=False)
              for phrase in phrases:
                flattenedForCosine.append((intent, phrase, training_phrases_with_embeddings[intent][phrase]))

            logger.info('Running cosine similarity for %s examples', len(flattenedForCosine))

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

            logger.info('Running cosine similarity for %s pairs of examples', len(workers))

            # data = Parallel(n_jobs=-1)(delayed(cosine_similarity_worker)(w[0], w[1], w[2], w[3], w[4], w[5]) for w in workers)
            data = [cosine_similarity_worker(w[0], w[1], w[2], w[3], w[4], w[5]) for w in workers]

            logger.info('Ready with cosine similarity for %s pairs, preparing results', len(data))

            similarity_df = pd.DataFrame(data, columns=['name1', 'example1', 'name2', 'example2', 'similarity'])
            similarity_different_intent = similarity_df['name1'] != similarity_df['name2']
            similarity_same_intent = similarity_df['name1'] == similarity_df['name2']

            similarity_different_intent_filtered = (similarity_df['name1'] != similarity_df['name2']) & (similarity_df['similarity'] > filter['minsimilarity'])
            similarity_df_sorted = similarity_df[similarity_different_intent_filtered].sort_values('similarity', ascending=False)
            similarity = [ { 'name1': name1, 'example1': example1, 'name2': name2, 'example2': example2, 'similarity': similarity } for name1, example1, name2, example2, similarity in zip(similarity_df_sorted['name1'], similarity_df_sorted['example1'], similarity_df_sorted['name2'], similarity_df_sorted['example2'], similarity_df_sorted['similarity'])]
            logger.debug(json.dumps(similarity, indent=2))

            cohesion_df_sorted = pd.DataFrame(similarity_df[similarity_same_intent].groupby('name1', as_index=False)['similarity'].mean()).sort_values('similarity', ascending=False)
            cohesion_df_sorted.columns = ['name', 'cohesion']
            cohesion = [ { 'name': name, 'cohesion': cohesion } for name, cohesion in zip(cohesion_df_sorted['name'], cohesion_df_sorted['cohesion'])]
            logger.debug(json.dumps(cohesion, indent=2))

            separation_df_sorted = pd.DataFrame(similarity_df[similarity_different_intent].groupby(['name1', 'name2'], as_index=False)['similarity'].mean()).sort_values('similarity', ascending=True)
            separation_df_sorted['separation'] = 1 - separation_df_sorted['similarity']
            separation = [ { 'name1': name1, 'name2': name2, 'separation': separation } for name1, name2, separation in zip(separation_df_sorted['name1'], separation_df_sorted['name2'], separation_df_sorted['separation'])]
            logger.debug(json.dumps(separation, indent=2))

            logger.info('Running chi2 analysis')
            #logger.info('Skipping chi2 analysis')

            flattenedForChi2 = pandas_utils.flatten_intents_list(intents)
            chi2, unigram_intent_dict, bigram_intent_dict = chi2_analyzer.get_chi2_analysis(flattenedForChi2, num_xgrams=filter['maxxgrams'])
            chi2_ambiguous_unigrams = chi2_analyzer.get_confusing_key_terms(unigram_intent_dict)
            chi2_ambiguous_bigrams = chi2_analyzer.get_confusing_key_terms(bigram_intent_dict)
            chi2_similarity = similarity_analyzer.ambiguous_examples_analysis(flattenedForChi2, filter['minsimilarity'])

            logger.info('Returning results')

            logger.info(f'Sending results to {boxEndpoint}')
            response_data = {
                "method": "calculate_embeddings",
                "status": "finished",
                "coachSessionId": coachSessionId,
                "output": {
                  'embeddings': embeddings_coords,
                  'similarity': similarity,
                  'cohesion': cohesion,
                  'separation': separation,
                  'chi2': [],
                  'chi2_ambiguous_unigrams': [],
                  'chi2_ambiguous_bigrams': [],
                  'chi2_similarity': []
                }
            }
            logger.debug(json.dumps(response_data, indent=2))
            res = requests.post(boxEndpoint, json = response_data)
            logger.info(res)
            calc_count += 1
        except Exception as e:
            logger.error('Calculating embeddings failed: ' + str(e))
            response_data = {
                "method": "calculate_embeddings",
                "status": "failed",
                "coachSessionId": coachSessionId,
                "err": 'Calculating embeddings failed: ' + str(e)
            }
            logger.debug(json.dumps(response_data, indent=2))
            res = requests.post(boxEndpoint, json = response_data)
            logger.info(res)

def ping():
  return 'Botium Coach Worker. Tensorflow Version: {tfVersion} PyTorch Version: {ptVersion}, Cuda: {ptCuda}'.format(
    tfVersion=tf.__version__, ptVersion=torch.__version__, ptCuda=str(torch.cuda.is_available()))

def calculate_embeddings(embeddingsRequest):
  with current_app.app_context():
      req_queue = current_app.req_queue
      req_queue.put(embeddingsRequest)

  coachSessionId = embeddingsRequest['coachSessionId']
  boxEndpoint = embeddingsRequest['boxEndpoint']

  return {
    'status': 'queued',
    'coachSessionId': coachSessionId,
    'boxEndpoint': boxEndpoint
  }

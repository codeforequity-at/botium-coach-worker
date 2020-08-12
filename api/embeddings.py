
import json
import logging
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Parallel, delayed
import math

logging.info('Loading word embeddings model from tfhub ...')
generate_embeddings = hub.load('https://tfhub.dev/google/universal-sentence-encoder-multilingual/3')
logging.info('Word embeddings model ready.')

def ping():
  return 'Botium Coach Worker. Tensorflow Version: {tfVersion}'.format(tfVersion=tf.__version__)

def cosine_similarity_worker(intent_1, phrase_1, embedd_1, intent_2, phrase_2, embedd_2):
  similarity = cosine_similarity([embedd_1], [embedd_2])[0][0]
  return [intent_1, phrase_1, intent_2, phrase_2, similarity]

def calculate_embeddings(embeddingsRequest):
  logging.debug(json.dumps(embeddingsRequest, indent=2))

  filter = embeddingsRequest['filter']
  intents = embeddingsRequest['intents']

  if len(intents) == 0:
    return { 'embeddings': [], 'similarity': [], 'cohesion': [], 'separation': [] }

  logging.info('Calculating embeddings for "%s" intents', len(intents))
  for intent in intents:
    logging.info('Calculating embeddings for intent "%s" with %s examples', intent['name'], len(intent['examples']))

  training_phrases_with_embeddings = defaultdict(list)
  for intent in intents:
    if len(intent['examples']) > 0:
      computed_embeddings = generate_embeddings(intent['examples'])
      training_phrases_with_embeddings[intent['name']] = dict(zip(intent['examples'], computed_embeddings))

  for intent_name, _ in training_phrases_with_embeddings.items():
    training_phrase, embeddings = next(iter(training_phrases_with_embeddings[intent_name].items()))
    logging.info('Calculated embeddings for intent {}, example: {{\'{}\':{}}}'.format(intent_name, training_phrase, embeddings[:5]))

  embedding_vectors = []

  for intent, training_phrases_and_embeddings in training_phrases_with_embeddings.items():
    for training_phrase, embeddings in training_phrases_and_embeddings.items():
      embedding_vectors.append(embeddings)

  embedding_vectors = np.asarray(embedding_vectors)

  logging.info('Starting principal component analysis')

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

  logging.debug(json.dumps(embeddings_coords, indent=2))

  flatten = []

  for intent in training_phrases_with_embeddings:
    for phrase in training_phrases_with_embeddings[intent]:
      flatten.append((intent, phrase, training_phrases_with_embeddings[intent][phrase]))

  logging.info('Ready with principal component analysis, running cosine similarity')

  workers = []
  for i in range(len(flatten)):
    for j in range(i+1, len(flatten)):

      intent_1 = flatten[i][0]
      phrase_1 = flatten[i][1]
      embedd_1 = flatten[i][2]

      intent_2 = flatten[j][0]
      phrase_2 = flatten[j][1]
      embedd_2 = flatten[j][2]

      workers.append((intent_1, phrase_1, embedd_1, intent_2, phrase_2, embedd_2))

  logging.info('Running cosine similarity for %s pairs of phrases', len(workers))

  # data = Parallel(n_jobs=-1)(delayed(cosine_similarity_worker)(w[0], w[1], w[2], w[3], w[4], w[5]) for w in workers)
  data = [cosine_similarity_worker(w[0], w[1], w[2], w[3], w[4], w[5]) for w in workers]

  logging.info('Ready with cosine similarity for %s pairs, preparing results', len(data))

  similarity_df = pd.DataFrame(data, columns=['name1', 'example1', 'name2', 'example2', 'similarity'])
  similarity_different_intent = similarity_df['name1'] != similarity_df['name2']
  similarity_same_intent = similarity_df['name1'] == similarity_df['name2']

  similarity_different_intent_filtered = (similarity_df['name1'] != similarity_df['name2']) & (similarity_df['similarity'] > filter['minsimilarity'])
  similarity_df_sorted = similarity_df[similarity_different_intent_filtered].sort_values('similarity', ascending=False)
  similarity = [ { 'name1': name1, 'example1': example1, 'name2': name2, 'example2': example2, 'similarity': similarity } for name1, example1, name2, example2, similarity in zip(similarity_df_sorted['name1'], similarity_df_sorted['example1'], similarity_df_sorted['name2'], similarity_df_sorted['example2'], similarity_df_sorted['similarity'])]
  logging.debug(json.dumps(similarity, indent=2))

  cohesion_df_sorted = pd.DataFrame(similarity_df[similarity_same_intent].groupby('name1', as_index=False)['similarity'].mean()).sort_values('similarity', ascending=False)
  cohesion_df_sorted.columns = ['name', 'cohesion']
  cohesion = [ { 'name': name, 'cohesion': cohesion } for name, cohesion in zip(cohesion_df_sorted['name'], cohesion_df_sorted['cohesion'])]
  logging.debug(json.dumps(cohesion, indent=2))

  separation_df_sorted = pd.DataFrame(similarity_df[similarity_different_intent].groupby(['name1', 'name2'], as_index=False)['similarity'].mean()).sort_values('similarity', ascending=True)
  separation_df_sorted['separation'] = 1 - separation_df_sorted['similarity']
  separation = [ { 'name1': name1, 'name2': name2, 'separation': separation } for name1, name2, separation in zip(separation_df_sorted['name1'], separation_df_sorted['name2'], separation_df_sorted['separation'])]
  logging.debug(json.dumps(separation, indent=2))

  logging.info('Returning results')

  return { 'embeddings': embeddings_coords, 'similarity': similarity, 'cohesion': cohesion, 'separation': separation }

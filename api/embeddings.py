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
import math

logging.info('Loading word embeddings model from tfhub ...')
generate_embeddings = hub.load('https://tfhub.dev/google/universal-sentence-encoder-multilingual/3')
logging.info('Word embeddings model ready.')

def ping():
  return 'Botium Coach Worker 1. Tensorflow Version: {tfVersion}'.format(tfVersion=tf.__version__)

def calculate_embeddings(intents):
  logging.debug(json.dumps(intents, indent=2))
  for intent in intents:
    logging.info('Calculating embeddings for intent "%s" with %s examples', intent['name'], len(intent['examples']))

  training_phrases_with_embeddings = defaultdict(list)
  for intent in intents:
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

  return { 'embeddings': embeddings_coords }

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import numpy as np
import pandas as pd
from collections import defaultdict
import os
import pickle
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity      
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
print('Tensorflow Version', tf.__version__)

def fetch_intents_training_phrases(directory):
  p = Path(directory)
  utterance_files = list(p.glob('**/*.utterances.txt'))
  intent_training_phrases = defaultdict(list)

  for utterance_file in utterance_files:
    with utterance_file.open(mode='r') as fid:
      lines = [line.strip() for line in fid]
      intent_training_phrases[lines[0]] = lines[1:]

  return intent_training_phrases

intent_training_phrases = fetch_intents_training_phrases('./data/Insurance3')

for intent in intent_training_phrases:
  print("{}:{}".format(intent, len(intent_training_phrases[intent])))

generate_embeddings = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

training_phrases_with_embeddings = defaultdict(list)
for intent_name, training_phrases_list in intent_training_phrases.items():
  computed_embeddings = generate_embeddings(training_phrases_list)
  training_phrases_with_embeddings[intent_name] = dict(zip(training_phrases_list, computed_embeddings))

for intent_name, _ in training_phrases_with_embeddings.items():
  training_phrase, embeddings = next(iter(training_phrases_with_embeddings[intent_name].items()))
  print("{}: {{'{}':{}}}".format(intent_name, training_phrase, embeddings[:5]))

embedding_vectors = []

for intent, training_phrases_and_embeddings in training_phrases_with_embeddings.items():
  for training_phrase, embeddings in training_phrases_and_embeddings.items():
    embedding_vectors.append(embeddings)

embedding_vectors = np.asarray(embedding_vectors)

pca = PCA(n_components=2)
pca.fit(embedding_vectors)

fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111)

legend = []

for color, intent in enumerate(training_phrases_with_embeddings.keys()):
  phrases = list(training_phrases_with_embeddings[intent].keys())
  embeddings = list(training_phrases_with_embeddings[intent].values())
  points = pca.transform(embeddings)
  xs = points[:,0]
  ys = points[:,1]
  ax.scatter(xs, ys, marker='o', s=100, c="C"+str(color))
  for i, phrase in enumerate(phrases):
    ax.annotate(phrase[:15] + '...', (xs[i], ys[i]))
  legend.append(intent)

ax.legend(legend)
plt.show()

flatten = []

for intent in training_phrases_with_embeddings:
  for phrase in training_phrases_with_embeddings[intent]:
    flatten.append((intent, phrase,  training_phrases_with_embeddings[intent][phrase]))

data = []
for i in range(len(flatten)):
  for j in range(i+1, len(flatten)):

    intent_1 = flatten[i][0]
    phrase_1 = flatten[i][1]
    embedd_1 = flatten[i][2]

    intent_2 = flatten[j][0]
    phrase_2 = flatten[j][1]
    embedd_2 = flatten[j][2]

    similarity = cosine_similarity([embedd_1], [embedd_2])[0][0]

    record = [intent_1, phrase_1, intent_2, phrase_2, similarity]
    data.append(record)

similarity_df = pd.DataFrame(data, 
  columns=["Intent A", "Phrase A", "Intent B", "Phrase B", "Similarity"])

different_intent = similarity_df['Intent A'] != similarity_df['Intent B']
print(similarity_df[different_intent].sort_values('Similarity', ascending=False).head(5))

same_intent = similarity_df['Intent A'] == similarity_df['Intent B']
cohesion_df = pd.DataFrame(similarity_df[different_intent].groupby('Intent A', as_index=False)['Similarity'].mean())
cohesion_df.columns = ['Intent', 'Cohesion']
print(cohesion_df)

different_intent = similarity_df['Intent A'] != similarity_df['Intent B']
separation_df = pd.DataFrame(similarity_df[different_intent].groupby(['Intent A', 'Intent B'], as_index=False)['Similarity'].mean())
separation_df['Separation'] = 1 - separation_df['Similarity']
del separation_df['Similarity']
print(separation_df.sort_values('Separation'))

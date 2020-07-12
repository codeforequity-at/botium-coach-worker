import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text

print('Downloading word embeddings model from tfhub ...')
hub.load('https://tfhub.dev/google/universal-sentence-encoder-multilingual/3')
print('Word embeddings model ready.')
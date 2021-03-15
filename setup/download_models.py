import tensorflow_hub as hub
import torch
from transformers import MBartForConditionalGeneration
# this import is fix because some strange bug coming on docker-build:
#   File "/usr/local/lib/python3.7/site-packages/tensorflow/python/framework/ops.py", line 3715, in _get_op_def
#     return self._op_def_cache[type]
# KeyError: 'SentencepieceOp'
import tensorflow_text

print('Downloading word embeddings model from tfhub ...')
hub.load('https://tfhub.dev/google/universal-sentence-encoder-multilingual/3')
print('Word embeddings model ready.')

print("PyTorch with cuda: " + str(torch.cuda.is_available()))

print('Downloading translation model for Huggingface Transformers ...')
MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
print('Translation model for Huggingface Transformers ready.')

from textattack.augmentation import EasyDataAugmenter, EmbeddingAugmenter, WordNetAugmenter

s = 'i want to book a flight to paris'

easy = EasyDataAugmenter()
print('EasyDataAugmenter:')
print(easy.augment(s))

emb = EmbeddingAugmenter()
print('EmbeddingAugmenter:')
for i in range(10):
  print(emb.augment(s))

wn = WordNetAugmenter()
print('WordNetAugmenter:')
for i in range(10):
  print(wn.augment(s))



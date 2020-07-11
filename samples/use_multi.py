import tensorflow_hub as hub
import numpy as np
import tensorflow_text

# Some texts of different lengths.
english_sentences = ["dog", "Puppies are nice.", "I enjoy taking long walks along the beach with my dog."]
italian_sentences = ["cane", "I cuccioli sono carini.", "Mi piace fare lunghe passeggiate lungo la spiaggia con il mio cane."]
german_sentences = ["hund", "welpen sind süß", "i mache gerne lange strandspaziergänge mit meinem hund."]

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

# Compute embeddings.
en_result = embed(english_sentences)
it_result = embed(italian_sentences)
de_result = embed(german_sentences)

# Compute similarity matrix. Higher score indicates greater similarity.
# similarity_matrix_it = np.inner(en_result, it_result)
similarity_matrix_de = np.inner(en_result, de_result)
print(similarity_matrix_de)
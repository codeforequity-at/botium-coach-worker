from pathlib import Path
from collections import defaultdict
from term_analysis import chi2_analyzer, similarity_analyzer
from utils import pandas_utils

def fetch_intents_training_phrases(directory):
  p = Path(directory)
  utterance_files = list(p.glob('**/*.utterances.txt'))
  intent_training_phrases = list()

  for utterance_file in utterance_files:
    with utterance_file.open(mode='r') as fid:
      lines = [line.strip() for line in fid]
      intent_training_phrases.append({ 'name': lines[0], 'examples': lines[1:] })

  return intent_training_phrases

intent_training_phrases = fetch_intents_training_phrases('../data/Banking')

flattened = pandas_utils.flatten_intents_list(intent_training_phrases)

print(flattened)

chi_df, unigram_intent_dict, bigram_intent_dict = chi2_analyzer.get_chi2_analysis(flattened)

print(chi_df)

ambiguous_unigram_df = chi2_analyzer.get_confusing_key_terms(unigram_intent_dict)
print(ambiguous_unigram_df)

ambiguous_bigram_df = chi2_analyzer.get_confusing_key_terms(bigram_intent_dict)
print(ambiguous_bigram_df)


#intent1 = 'Goodbye'
#intent2 = ''
#chi2_analyzer.chi2_overlap_check(ambiguous_unigram_df,ambiguous_bigram_df,intent1,intent2)

similar_utterance_diff_intent_pd = similarity_analyzer.ambiguous_examples_analysis(flattened)
print(similar_utterance_diff_intent_pd)


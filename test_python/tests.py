import unittest
import pickle
from api.term_analysis import chi2_analyzer, similarity_analyzer
import logging

class TestChi2Calculation(unittest.TestCase):

    def test_flattened_for_chi2(self):
        fileObj = open('test_data/flattenedForChi2.obj', 'rb')
        flattenedForChi2_test = pickle.load(fileObj)
        fileObj.close()
        fileObj = open('test_data/chi2.obj', 'rb')
        chi2_test = pickle.load(fileObj)
        fileObj.close()
        fileObj = open('test_data/unigram_intent_dict.obj', 'rb')
        unigram_intent_dict_test = pickle.load(fileObj)
        fileObj.close()
        fileObj = open('test_data/bigram_intent_dict.obj', 'rb')
        bigram_intent_dict_test = pickle.load(fileObj)
        fileObj.close()

        chi2, unigram_intent_dict, bigram_intent_dict = chi2_analyzer.get_chi2_analysis(
                                logging, False, '', flattenedForChi2_test, num_xgrams=8)

        self.assertEqual(chi2, chi2_test)
        self.assertEqual(unigram_intent_dict, unigram_intent_dict_test)
        self.assertEqual(bigram_intent_dict, bigram_intent_dict_test)

    def test_get_confusing_key_terms_unigrams(self):
        fileObj = open('test_data/chi2_ambiguous_unigrams.obj', 'rb')
        chi2_ambiguous_unigrams_test = pickle.load(fileObj)
        fileObj.close()
        fileObj = open('test_data/unigram_intent_dict.obj', 'rb')
        unigram_intent_dict_test = pickle.load(fileObj)
        fileObj.close()

        chi2_ambiguous_unigrams = chi2_analyzer.get_confusing_key_terms(
                        unigram_intent_dict_test)

        self.assertEqual(chi2_ambiguous_unigrams, chi2_ambiguous_unigrams_test)

    def test_get_confusing_key_terms_bigrams(self):
        fileObj = open('test_data/chi2_ambiguous_bigrams.obj', 'rb')
        chi2_ambiguous_bigrams_test = pickle.load(fileObj)
        fileObj.close()
        fileObj = open('test_data/bigram_intent_dict.obj', 'rb')
        bigram_intent_dict_test = pickle.load(fileObj)
        fileObj.close()

        chi2_ambiguous_bigrams = chi2_analyzer.get_confusing_key_terms(
                        bigram_intent_dict_test)

        self.assertEqual(chi2_ambiguous_bigrams, chi2_ambiguous_bigrams_test)

    def test_ambiguous_examples_analysis(self):
        fileObj = open('test_data/chi2_similarity.obj', 'rb')
        chi2_similarity_test = pickle.load(fileObj)
        fileObj.close()
        fileObj = open('test_data/flattenedForChi2.obj', 'rb')
        flattenedForChi2_test = pickle.load(fileObj)
        fileObj.close()

        chi2_similarity = similarity_analyzer.ambiguous_examples_analysis(
                        logging, False, '', flattenedForChi2_test, 0.5)

        self.assertEqual(chi2_similarity, chi2_similarity_test)
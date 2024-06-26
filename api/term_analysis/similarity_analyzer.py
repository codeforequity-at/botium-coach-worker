import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from concurrent.futures import ThreadPoolExecutor, as_completed

def ambiguous_examples_analysis(logger, log_extras, worker_name, workspace_pd, threshold=0.7):
    """
    Analyze the test workspace and find out similar utterances that belongs to different intent
    :param workspace_pd: pandas dataframe in format of [utterance,label]
    :param threshold: cut off for similarity score
    :return: pands dataframe in format of ['Intent1', 'Utterance1', 'Intent2', 'Utterance2',
                                           'similarity score']
    """

    logger.info('chi2 similarity: create the feature matrix')
    # first create the feature matrix
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    workspace_bow = vectorizer.fit_transform(workspace_pd["utterance"]).todense()

    logger.info('chi2 similarity: calculate_cosine_similarity', extra=log_extras)
    cos_sim_score_matrix = _calculate_cosine_similarity(logger, log_extras, worker_name, workspace_bow)

    logger.info('chi2 similarity: remove the lower triangle of the matrix and apply threshold')
    # remove the lower triangle of the matrix and apply threshold
    similar_utterance_index = np.argwhere(
        (cos_sim_score_matrix - np.tril(cos_sim_score_matrix)) > threshold
    )

    logger.info('chi2 similarity: post processing', extra=log_extras)
    temp_pds = []
    for index in similar_utterance_index:
        if (
            workspace_pd["intent"].iat[index[0]]
            != workspace_pd["intent"].iat[index[1]]
        ):
            intent1 = workspace_pd["intent"].iat[index[0]]
            utterance1 = workspace_pd["utterance"].iat[index[0]]
            intent2 = workspace_pd["intent"].iat[index[1]]
            utterance2 = workspace_pd["utterance"].iat[index[1]]
            score = cos_sim_score_matrix[index[0], index[1]]
            temp_pd = {
                "name1": [intent1],
                "example1": [utterance1],
                "name2": [intent2],
                "example2": [utterance2],
                "similarity": [score],
            }
            temp_pds.append(temp_pd)
    similar_utterance_pd = pd.DataFrame(temp_pds, columns=["name1", "example1", "name2", "example2", "similarity"])

    logger.info('chi2 similarity: sorting by similarity', extra=log_extras)
    similarity_df_sorted = similar_utterance_pd.sort_values(by=["similarity"], ascending=False)
    logger.info('chi2 similarity: sorting by similarity done', extra=log_extras)
    return [ { 'name1': name1, 'example1': example1, 'name2': name2, 'example2': example2, 'similarity': similarity } for name1, example1, name2, example2, similarity in zip(similarity_df_sorted['name1'], similarity_df_sorted['example1'], similarity_df_sorted['name2'], similarity_df_sorted['example2'], similarity_df_sorted['similarity'])]

def _calculate_cosine_similarity(logger, log_extras, worker_name, workspace_bow):
    """
    Given bow representation of the workspace utterance, calculate cosine similarity score
    :param workspace_bow: dense representation of BOW of workspace utterances
    :return: cosine_similarity_matrix
    """
    # normalized and calculate cosine similarity
    logger.info('Calculating cosine similarity: normalizing ...', extra=log_extras)
    workspace_bow = np.asarray(workspace_bow, np.float32)
    workspace_bow = workspace_bow / np.linalg.norm(workspace_bow, axis=1, keepdims=True)
    logger.info('Calculating cosine similarity: dot product ...', extra=log_extras)
    cosine_similarity_matrix = workspace_bow.dot(np.transpose(workspace_bow))
    logger.info('Calculating cosine similarity done ...', extra=log_extras)
    return cosine_similarity_matrix

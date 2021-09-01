import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

def ambiguous_examples_analysis(logger, workspace_pd, threshold=0.7):
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

    logger.info('chi2 similarity: calculate_cosine_similarity')
    cos_sim_score_matrix = _calculate_cosine_similarity(logger, workspace_bow)

    logger.info('chi2 similarity: remove the lower triangle of the matrix and apply threshold')
    # remove the lower triangle of the matrix and apply threshold
    similar_utterance_index = np.argwhere(
        (cos_sim_score_matrix - np.tril(cos_sim_score_matrix)) > threshold
    )
    similar_utterance_pd = pd.DataFrame(
        columns=["name1", "example1", "name2", "example2", "similarity"]
    )

    logger.info('chi2 similarity: post processing')
    for index in similar_utterance_index:
        if (
            workspace_pd["intent"].iloc[index[0]]
            != workspace_pd["intent"].iloc[index[1]]
        ):
            intent1 = workspace_pd["intent"].iloc[index[0]]
            utterance1 = workspace_pd["utterance"].iloc[index[0]]
            intent2 = workspace_pd["intent"].iloc[index[1]]
            utterance2 = workspace_pd["utterance"].iloc[index[1]]
            score = cos_sim_score_matrix[index[0], index[1]]
            temp_pd = pd.DataFrame(
                {
                    "name1": [intent1],
                    "example1": [utterance1],
                    "name2": [intent2],
                    "example2": [utterance2],
                    "similarity": [score],
                }
            )
            similar_utterance_pd = similar_utterance_pd.append(
                temp_pd, ignore_index=True
            )

    logger.info('chi2 similarity: sorting by similarity')
    similarity_df_sorted = similar_utterance_pd.sort_values(by=["similarity"], ascending=False)
    return [ { 'name1': name1, 'example1': example1, 'name2': name2, 'example2': example2, 'similarity': similarity } for name1, example1, name2, example2, similarity in zip(similarity_df_sorted['name1'], similarity_df_sorted['example1'], similarity_df_sorted['name2'], similarity_df_sorted['example2'], similarity_df_sorted['similarity'])]

def _calculate_cosine_similarity(logger, workspace_bow):
    """
    Given bow representation of the workspace utterance, calculate cosine similarity score
    :param workspace_bow: dense representation of BOW of workspace utterances
    :return: cosine_similarity_matrix
    """
    # normalized and calculate cosine similarity
    logger.info('1')
    workspace_bow = np.asarray(workspace_bow, np.float32)
    workspace_bow = workspace_bow / np.linalg.norm(workspace_bow, axis=1, keepdims=True)
    logger.info('2')
    cosine_similarity_matrix = workspace_bow.dot(np.transpose(workspace_bow))
    logger.info('3')
    return cosine_similarity_matrix

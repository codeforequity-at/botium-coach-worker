import openai
import os
import pinecone

from flask import current_app
from .utils.log import getLogger
from .utils.factcheck import editor, document_upsert_pinecone, pinecone_init, create_sample_questions

logger = getLogger('Fact Checker')

def create_index(CreateIndexRequest):
    """
        Creates a Pinecone index to upload embeddings to

        inputs: index (string) - name specified to call index on pinecone
                environment (string) - pincone environment where index is stored

        output: result - Dict with 2 keys: 
                        status  - confirms if index was successfully created or not (True/False)
                        message - contains string stating if it was successfully created or failed with failure message
    """
    index = CreateIndexRequest['index']
    pine_api_key = os.environ.get('PINECONE_API')
    pine_env = CreateIndexRequest['environment']
    try:
        pinecone.init(api_key=pine_api_key, environment=pine_env)
        active_indexes = pinecone.list_indexes()
        if index in active_indexes:
            return {
                'status': "finished",
                'message': f'Index {index} in environment {pine_env} already active'
            }

        pinecone.create_index(index, dimension=1536,
                              metric='cosine', pods=1, replicas=1)
        logger.info(
            f'Created Pinecone index {index} in environment {pine_env}')
        return {
            'status': "finished",
            'message': f'Successfully created index {index} in environment {pine_env}'
        }
    except Exception as error:
        logger.error(
            f'Creating Pinecone index {index} in environment {pine_env} failed: {format(error)}')
        # handle the exception
        return {
            'status': "failed",
            'err': f'Creating index {index} in environment {pine_env} failed: {format(error)}'
        }


def upload_factcheck_documents_worker(logger, worker_name, req_queue, res_queue, err_queue, UploadFactcheckDocumentRequest):
    pine_api_key = os.environ.get('PINECONE_API')
    openai.api_key = os.environ.get('OPEN_API')
    embedding_model = "text-embedding-ada-002"

    sessionId = UploadFactcheckDocumentRequest['factcheckSessionId']
    index = UploadFactcheckDocumentRequest['index']
    pine_env = UploadFactcheckDocumentRequest['environment']
    namespace = UploadFactcheckDocumentRequest.get('namespace', None)
    documents = UploadFactcheckDocumentRequest['documents']

    response_data = {}
    if 'boxEndpoint' in UploadFactcheckDocumentRequest:
        response_data['boxEndpoint'] = UploadFactcheckDocumentRequest['boxEndpoint']
        response_data['header'] = {"content-type": "application/json"}

    response_data['redisKey'] = 'coachworker_res_factcheckupload_' + sessionId
    response_data['deleteRedisKey'] = 'coachworker_req_factcheckupload_' + sessionId

    current_filename = None

    try:
        pineindex = pinecone_init(pine_api_key, pine_env, index)

        logger.info(f'Deleting vectors in Pinecone index {index} in environment {pine_env}/{namespace}')
        pineindex.delete(delete_all=True, namespace=namespace)

        for document in documents:
            current_filename = document["filename"]
            content = document_upsert_pinecone(
                openai, embedding_model, pineindex, namespace, current_filename, document["text"])
            logger.info(
                f'Uploading {current_filename} to Pinecone index {index} in environment {pine_env}/{namespace}: {content["message"]}')

        response_data['json'] = {
            "method": "upload_factcheck_documents",
            "status": "finished",
            "factcheckSessionId": sessionId
        }
        res_queue.put((response_data,))
    except Exception as error:
        logger.error(
            f'Uploading {current_filename} to Pinecone index {index} in environment {pine_env}/{namespace} failed: {format(error)}')
        response_data['json'] = {
            "method": "upload_factcheck_documents",
            "status": "failed",
            "factcheckSessionId": sessionId,
            "err": f'Uploading {current_filename} to index {index} in environment {pine_env}/{namespace} failed: {format(error)}'
        }
        res_queue.put((response_data,))


def upload_factcheck_documents(UploadFactcheckDocumentRequest):
    sessionId = UploadFactcheckDocumentRequest['factcheckSessionId']

    with current_app.app_context():
        req_queue = current_app.req_queue
        req_queue.put((UploadFactcheckDocumentRequest,
                      "upload_factcheck_documents"))

    return {
        'status': 'queued',
        'message': "Started uploading documents to index.",
        'factcheckSessionId': sessionId
    }


def create_sample_queries_worker(logger, worker_name, req_queue, res_queue, err_queue, CreateFactcheckSampleQueriesRequest):
    openai.api_key = os.environ.get('OPEN_API')

    sessionId = CreateFactcheckSampleQueriesRequest['factcheckSessionId']
    documents = CreateFactcheckSampleQueriesRequest['documents']

    response_data = {}
    if 'boxEndpoint' in CreateFactcheckSampleQueriesRequest:
        response_data['boxEndpoint'] = CreateFactcheckSampleQueriesRequest['boxEndpoint']
        response_data['header'] = {"content-type": "application/json"}

    response_data['redisKey'] = 'coachworker_res_createsamplequeries_' + sessionId
    response_data['deleteRedisKey'] = 'coachworker_req_createsamplequeries_' + sessionId

    current_filename = None

    try:
        sample_queries = []
        for document in documents:
            current_filename = document["filename"]
            questions = create_sample_questions(openai, document["text"])
            logger.info(
                f'Created sample queries for {current_filename}: {questions}')
            sample_queries = sample_queries + questions

        response_data['json'] = {
            "method": "create_sample_queries",
            "status": "finished",
            "factcheckSessionId": sessionId,
            "sample_queries": sample_queries
        }
        res_queue.put((response_data,))
    except Exception as error:
        logger.error(
            f'Creating sample queries for {current_filename} failed: {format(error)}')
        response_data['json'] = {
            "method": "create_sample_queries",
            "status": "failed",
            "factcheckSessionId": sessionId,
            "err": f'Creating sample queries for {current_filename} failed: {format(error)}'
        }
        res_queue.put((response_data,))


def create_sample_queries(CreateFactcheckSampleQueriesRequest):
    sessionId = CreateFactcheckSampleQueriesRequest['factcheckSessionId']

    with current_app.app_context():
        req_queue = current_app.req_queue
        req_queue.put((CreateFactcheckSampleQueriesRequest,
                      "create_sample_queries"))

    return {
        'status': 'queued',
        'message': "Started creating sample queries.",
        'factcheckSessionId': sessionId
    }


def factcheck(factcheckRequest):
    """
        Fact checks a statment given the ground truth docs stored on pinecone index

        inputs: index (string) - name of index where embeddings of ground truth docs are stored on pinecone
                environment (string) - pincone environmnet where index is stored
                statement (string) - the statement to be fact checked  

        output: Dictionary with 3 keys: 
                status - confirms if statement is factually correct or not True/False)
                reasoning - contains reasons why we beleive statement is true or false
                fixed_statement - edited version of statement based on reasons 
    """
    pine_api_key = os.environ.get('PINECONE_API')
    openai.api_key = os.environ.get('OPEN_API')

    index = factcheckRequest['index']
    pine_env = factcheckRequest['environment']
    namespace = factcheckRequest.get('namespace', None)
    statement = factcheckRequest['statement']

    pineindex = pinecone_init(pine_api_key, pine_env, index)
    editor_responses, agreement_gates, status = editor(
        openai, statement, pineindex, namespace)

    result = {'status': status,
              'reasoning': agreement_gates,
              'fixed_statement': editor_responses}
    return result

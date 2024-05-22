import openai
import os
from pinecone import Pinecone, PodSpec

from flask import current_app
from .utils.log import getLogger
from .utils.factcheck import editor, document_upsert_pinecone, pinecone_init, create_sample_questions

import gevent


logger = getLogger('fact_checker')
createIndexLogger = getLogger(f'fact_checker.create_index')
deleteLogger = getLogger(f'fact_checker.delete_factcheck_documents')
factCheckLogger = getLogger(f'fact_checker.factcheck')

pine_api_key = os.environ.get('PINECONE_API')
pine_environment = os.environ.get('PINECONE_ENVIRONMENT')
pine_index = os.environ.get('PINECONE_INDEX')
openai.api_key = os.environ.get('OPEN_API')

def create_index2(CreateIndexRequest):
    index = CreateIndexRequest.get('index', pine_index)
    pine_env = CreateIndexRequest.get('environment', pine_environment)
    try:
        pc = Pinecone(api_key=pine_api_key, environment=pine_env)
        active_indexes = pc.list_indexes().names()
        if index in active_indexes:
            return {
                'status': "finished",
                'message': f'Index {index} in environment {pine_env} already active'
            }

        pc.create_index(
            name=index,
            dimension=1536,
            metric='cosine',
            spec=PodSpec(
              environment=pine_env,
              pods=1,
              replicas=1
            )
        )
        createIndexLogger.info(f'Created Pinecone index {index} in environment {pine_env}')
        return {
            'status': "finished",
            'message': f'Successfully created index {index} in environment {pine_env}'
        }
    except Exception as error:
        createIndexLogger.error(f'Creating Pinecone index {index} in environment {pine_env} failed: {str(error)}')
        # handle the exception
        return {
            'status': "failed",
            'err': f'Creating index {index} in environment {pine_env} failed: {str(error)}'
        }

def create_index(CreateIndexRequest):
    t = gevent.joinall([gevent.spawn(create_index2, CreateIndexRequest)])
    print (t[0].value)
    return t[0].value

def upload_factcheck_documents_worker(logger, worker_name, req_queue, res_queue, err_queue, UploadFactcheckDocumentRequest):
    embedding_model = "text-embedding-ada-002"

    sessionId = UploadFactcheckDocumentRequest['factcheckSessionId']
    index = UploadFactcheckDocumentRequest.get('index', pine_index)
    pine_env = UploadFactcheckDocumentRequest.get('environment', pine_environment)
    namespace = UploadFactcheckDocumentRequest.get('namespace', None)
    documents = UploadFactcheckDocumentRequest['documents']

    response_data = {}
    if 'boxEndpoint' in UploadFactcheckDocumentRequest:
        response_data['boxEndpoint'] = UploadFactcheckDocumentRequest['boxEndpoint']
        response_data['header'] = {"content-type": "application/json"}

    current_filename = None

    try:
        pineindex = pinecone_init(logger, pine_api_key, pine_env, index)

        logger.info(f'Deleting vectors in Pinecone index {index} in environment {pine_env}/{namespace}')
        try:
            pineindex.delete(delete_all=True, namespace=namespace)
        except Exception as error:
            logger.warn(f'Cleaning namespace in Pinecone index {index} in environment {pine_env}/{namespace} failed: {str(error)}')

        for document in documents:
            current_filename = document["filename"]
            content = document_upsert_pinecone(logger,
                openai, embedding_model, pineindex, namespace, current_filename, document["text"])
            logger.info(f'Uploading {current_filename} to Pinecone index {index} in environment {pine_env}/{namespace}: {content["message"]}')

        response_data['json'] = {
            "method": "upload_factcheck_documents",
            "status": "finished",
            "factcheckSessionId": sessionId
        }
        res_queue.put((response_data,))
    except Exception as error:
        logger.error(f'Uploading {current_filename} to Pinecone index {index} in environment {pine_env}/{namespace} failed: {str(error)}')
        response_data['json'] = {
            "method": "upload_factcheck_documents",
            "status": "failed",
            "factcheckSessionId": sessionId,
            "err": f'Uploading {current_filename} to index {index} in environment {pine_env}/{namespace} failed: {str(error)}'
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

def delete_factcheck_documents2(DeleteFactcheckDocumentRequest):
    index = DeleteFactcheckDocumentRequest.get('index', pine_index)
    pine_env = DeleteFactcheckDocumentRequest.get('environment', pine_environment)
    namespace = DeleteFactcheckDocumentRequest.get('namespace', None)

    if namespace is None or namespace == "":
        return {
            'status': "finished",
            'message': f'Namespace not given for index {index} in environment {pine_env}'
        }

    try:
        pineindex = pinecone_init(deleteLogger, pine_api_key, pine_env, index)   

        if namespace in pineindex.describe_index_stats()['namespaces'].keys():
            pineindex.delete(delete_all=True, namespace=namespace)
            deleteLogger.info(f'Deleted namespace {namespace} in Pinecone index {index} in environment {pine_env}')
            return {
                'status': "finished",
                'message': f'Successfully deleted namespace {namespace} in index {index} in environment {pine_env}'
            }
        else:
            return {
                'status': "finished",
                'message': f'No Namespace {namespace} found in index {index} in environment {pine_env}'
            }
    except Exception as error:
        deleteLogger.error(f'Deleting namespace {namespace} in Pinecone index {index} in environment {pine_env} failed: {str(error)}')
        # handle the exception
        return {
            'status': "failed",
            'err': f'Deleting namespace {namespace} in index {index} in environment {pine_env} failed: {str(error)}'
        }

def delete_factcheck_documents(DeleteFactcheckDocumentRequest):
    t = gevent.joinall([gevent.spawn(delete_factcheck_documents2, DeleteFactcheckDocumentRequest)])
    print (t[0].value)
    return t[0].value

def create_sample_queries_worker(logger, worker_name, req_queue, res_queue, err_queue, CreateFactcheckSampleQueriesRequest):
    sessionId = CreateFactcheckSampleQueriesRequest['factcheckSessionId']
    documents = CreateFactcheckSampleQueriesRequest['documents']

    response_data = {}
    if 'boxEndpoint' in CreateFactcheckSampleQueriesRequest:
        response_data['boxEndpoint'] = CreateFactcheckSampleQueriesRequest['boxEndpoint']
        response_data['header'] = {"content-type": "application/json"}

    current_filename = None

    try:
        sample_queries = []
        for document in documents:
            current_filename = document["filename"]
            questions = create_sample_questions(openai, document["text"])
            logger.info(f'Created sample queries for {current_filename}: {questions}')
            sample_queries = sample_queries + questions

        response_data['json'] = {
            "method": "create_sample_queries",
            "status": "finished",
            "factcheckSessionId": sessionId,
            "sample_queries": sample_queries
        }
        res_queue.put((response_data,))
    except Exception as error:
        logger.error(f'Creating sample queries for {current_filename} failed: {str(error)}')
        response_data['json'] = {
            "method": "create_sample_queries",
            "status": "failed",
            "factcheckSessionId": sessionId,
            "err": f'Creating sample queries for {current_filename} failed: {str(error)}'
        }
        res_queue.put((response_data,))


def create_sample_queries2(CreateFactcheckSampleQueriesRequest):
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

def create_sample_queries(CreateFactcheckSampleQueriesRequest):
    t = gevent.joinall([gevent.spawn(create_sample_queries_worker, logger, 'create_sample_queries', current_app.req_queue, current_app.res_queue, current_app.err_queue, CreateFactcheckSampleQueriesRequest)])
    print (t[0].value)
    return t[0].value


def factcheck2(factcheckRequest):
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


    index = factcheckRequest.get('index', pine_index)
    pine_env = factcheckRequest.get('environment', pine_environment)
    namespace = factcheckRequest.get('namespace', None)
    statement = factcheckRequest['statement']

    pineindex = pinecone_init(factCheckLogger, pine_api_key, pine_env, index)

    editor_responses, agreement_gates, status = editor(factCheckLogger, openai, statement, pineindex, namespace)

    result = {'status': status,
              'reasoning': agreement_gates,
              'fixed_statement': editor_responses}
    return result

def factcheck(factcheckRequest):
    t = gevent.joinall([gevent.spawn(factcheck2, factcheckRequest)])
    print (t[0].value)
    return t[0].value
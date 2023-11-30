from ast import Return
import json
import math
import numpy as np
import openai
import os
import pandas as pd
import pickle
import pinecone
import PyPDF2
import re
import requests
import sys
import time
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_text
import torch
import multiprocessing as mp
import uuid

from api.term_analysis import chi2_analyzer, similarity_analyzer
from api.utils import pandas_utils
from enum import Enum
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import current_app
from flask_healthz import healthz
from functools import singledispatch
from joblib import Parallel, delayed
from .redis_client import getRedis
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from .utils.log import getLogger
from .utils.factcheck import editor, document_upsert_pinecone, pinecone_init

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
    try :
        pinecone.init(api_key=pine_api_key, environment=pine_env)
        active_indexes = pinecone.list_indexes()
        if index in active_indexes:
          return {
            'status': True,
            'message': f'Index {index} in environment {pine_env} already active'
          }

        pinecone.create_index(index, dimension=1536, metric='cosine', pods=1, replicas=1)
        logger.info(f'Created Pinecone index {index} in environment {pine_env}')
        return {
          'status': True,
          'message': f'Successfully created index {index} in environment {pine_env}'
        }
    except Exception as error:
        logger.error(f'Creating Pinecone index {index} in environment {pine_env} failed: {format(error)}')
        # handle the exception
        return {
          'status': False,
          'message': f'Creating index {index} in environment {pine_env} failed: {format(error)}'
        }

def upload_factcheck_documents_worker(logger, worker_name, req_queue, res_queue, err_queue, UploadFactcheckDocumentRequest):
    """
        Uploads embeddings to Pinecone index.

        inputs: index (string) - name of pinecone index to store embeddings
                environment (string) - pinecone environment where index is stored
                namespace (string) - pinecone namespace
                filepath (string) - filepath of where documents to be uploaded are stored

        output: content - Dict with 2 keys: 
                        status - confirms if index was successfully uploaded or not (True/False)
                        message - contains string stating if documents were sucessfully uploaded or failed with failure message.
    """
    pine_api_key = os.environ.get('PINECONE_API')
    openai.api_key = os.environ.get('OPEN_API')
    embedding_model = "text-embedding-ada-002"

    sessionId = UploadFactcheckDocumentRequest['factcheckSessionId']
    index = UploadFactcheckDocumentRequest['index']
    pine_env = UploadFactcheckDocumentRequest['environment']
    namespace = UploadFactcheckDocumentRequest['namespace']
    filepath = UploadFactcheckDocumentRequest['filepath']

    response_data = {}
    if 'boxEndpoint' in UploadFactcheckDocumentRequest:
        response_data['boxEndpoint'] = UploadFactcheckDocumentRequest['boxEndpoint']
        response_data['header'] = { "content-type": "application/json" }

    response_data['redisKey'] = 'coachworker_res_factcheckupload_' + sessionId
    response_data['deleteRedisKey'] = 'coachworker_req_factcheckupload_' + sessionId

    try:
        pineindex = pinecone_init(pine_api_key,pine_env,index)
        content = document_upsert_pinecone(openai, embedding_model, pineindex, namespace, filepath)
        logger.info(content['message'])
        response_data['json'] = {
            "method": "upload_factcheck_documents",
            "status": "finished",
            "factcheckSessionId": sessionId,
            "content": content
        }
        res_queue.put((response_data,))
    except Exception as error:
        logger.error(f'Uploading to Pinecone index {index} in environment {pine_env}/{namespace} failed: {format(error)}')
        response_data['json'] = {
            "method": "upload_factcheck_documents",
            "status": "failed",
            "factcheckSessionId": sessionId,
            "err": f'Uploading to index {index} in environment {pine_env}/{namespace} failed: {format(error)}'
        }
        res_queue.put((response_data,))

def upload_factcheck_documents(UploadFactcheckDocumentRequest):
    sessionId = UploadFactcheckDocumentRequest['factcheckSessionId']

    with current_app.app_context():
        req_queue = current_app.req_queue
        req_queue.put((UploadFactcheckDocumentRequest, "upload_factcheck_documents"))

    return {
      'status': 'queued',
      'message': "Started uploading documents to index.",
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
    statement = factcheckRequest['statement']

    pineindex=pinecone_init(pine_api_key,pine_env,index)
    editor_responses, agreement_gates, status =editor(openai, statement, pineindex, index)
   
    result = {  'status': status,
                'reasoning': agreement_gates,
                'fixed_statement': editor_responses}
    return result
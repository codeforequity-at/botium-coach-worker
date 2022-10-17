#!/usr/bin/env python3
import os
import connexion
import logging
from flask import current_app
import multiprocessing as mp
from api.embeddings import calculate_embeddings_worker
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Parallel, delayed
from api.term_analysis import chi2_analyzer, similarity_analyzer
from api.utils import pandas_utils
from multiprocessing import Process, RawValue, Lock
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import json
import numpy as np
import pandas as pd
import math
from collections import defaultdict
import torch
import requests
import gc
import inspect
import redis

def redis_scheduler(req_queue,log_format,log_level,log_datefmt):
    in_queue = []
    red = redis.Redis(host='localhost', port=6379, db=0) 
    while True:
        for k in red.scan_iter("coachworker_req*"):
            #print(red.get(k))
            if k not in in_queue:
                if k.decode("utf-8").startswith('coachworker_req_chi2'):
                    req_queue.put((json.loads(red.get(k)), "calculate_chi2"))
                if k.decode("utf-8").startswith('coachworker_req_embeddings'):
                    req_queue.put((json.loads(red.get(k)), "calculate_embeddings"))
                in_queue.append(k)

def process_scheduler(req_queue,log_format,log_level,log_datefmt):
    logger = logging.getLogger('Worker scheduler')
    logger.setLevel(log_level)
    logger.info('Worker scheduler started...')
    processes = []
    for i in range(int(os.environ.get('COACH_PARALLEL_WORKERS', 1))):
        p = mp.Process(target=calculate_embeddings_worker, args=(req_queue,i,log_format,log_level,log_datefmt))
        p.daemon = False
        p.start()
        processes.append(p)
    while True:
        for i in range(len(processes)):
            p = processes[i]
            if not p.is_alive():
                p = mp.Process(target=calculate_embeddings_worker, args=(req_queue,i,log_format,log_level,log_datefmt))
                p.daemon = False
                p.start()
                processes[i] = p

def create_app():
    LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
    log_datefmt = '%Y-%m-%d %H:%M:%S'
    log_level = LOGLEVEL
    log_format = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=log_format, level=log_level, datefmt=log_datefmt)
    app = connexion.App(__name__, specification_dir='openapi/')
    app.add_api('botium_coach_worker_api.yaml')
    req_queue = mp.Queue()
    p = mp.Process(target=process_scheduler, args=(req_queue,log_format,log_level,log_datefmt))
    p.start()
    p = mp.Process(target=redis_scheduler, args=(req_queue,log_format,log_level,log_datefmt))
    p.start()
    with app.app.app_context():
        current_app.req_queue = req_queue

    return app

if os.environ.get('GUNICORN_MODE', 0) == 0:
    if __name__ == '__main__':
      app = create_app()
      app.run(port=int(os.environ.get('PORT', 4444)))
else:
    app = create_app()

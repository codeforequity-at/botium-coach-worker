#!/usr/bin/env python3
import os
import connexion
from functools import singledispatch
from flask import current_app
from flask_healthz import healthz
import multiprocessing as mp
from api.embeddings import calculate_embeddings_worker
from api.factcheck import upload_factcheck_documents_worker, create_sample_queries_worker
from multiprocessing import Process, RawValue, Lock
import json
import time
import requests
import numpy as np
from api.utils.log import getLogger
from api.redis_client import getRedis


redis_enabled = int(os.environ.get('REDIS_ENABLE', 0)) == 1
max_retries = int(os.environ.get('COACH_RETRY_REQUEST_RETRIES', 12))
retry_delay_seconds = int(os.environ.get('COACH_RETRY_REQUEST_DELAY', 10))
maxCalcCount = int(os.environ.get('COACH_MAX_CALCULATIONS_PER_WORKER', 100))

def process_redis(req_queue, res_queue, err_queue):
    logger = getLogger('process_redis')

    in_queue = []
    red = getRedis()

    logger.info('Worker process_redis started...')
    while True:
        for k in red.scan_iter("coachworker_req*"):
            if k not in in_queue:
                if k.decode("utf-8").startswith('coachworker_req_chi2'):
                    req_obj = json.loads(red.get(k))
                    req_queue.put((req_obj, "calculate_chi2"))
                    res_queue.put(({
                      'redisKey': 'coachworker_status_chi2_' + req_obj['coachSessionId'],
                      'data': {
                        "method": "calculate_chi2",
                        "clientId": req_obj['clientId'],
                        "coachSessionId": req_obj['coachSessionId'],
                        "status": 'QUEUED',
                        'statusDescription': 'Request for Chi2 Analysis is queued'
                      }
                    },))
                if k.decode("utf-8").startswith('coachworker_req_embeddings'):
                    req_obj = json.loads(red.get(k))
                    req_queue.put((req_obj, "calculate_embeddings"))
                    res_queue.put(({
                      'redisKey': 'coachworker_status_embeddings_' + req_obj['coachSessionId'],
                      'data': {
                        "method": "calculate_embeddings",
                        "clientId": req_obj['clientId'],
                        "coachSessionId": req_obj['coachSessionId'],
                        "status": 'QUEUED',
                        'statusDescription': 'Request for Embeddings Analysis is queued'
                      }
                    },))
                in_queue.append(k)
        time.sleep(5)

@singledispatch
def to_serializable(val):
    """Used by default."""
    return str(val)

@to_serializable.register(np.float32)
def ts_float32(val):
    """Used if *val* is an instance of numpy.float32."""
    return np.float64(val)

def process_responses(req_queue, res_queue, err_queue):
    logger = getLogger('process_responses')

    red = None
    if redis_enabled:
        red = getRedis()

    logger.info('Worker process_responses started...')
    while True:
        response_data, retryCount, retryMethod = (lambda a, b=None, c=None: (a, b, c))(*res_queue.get())

        if retryCount is None:
            retryCount = max_retries

        if retryMethod is None:
            if redis_enabled and 'redisKey' in response_data:
                res_queue.put((response_data, retryCount, 'retryRedis'))
            if 'boxEndpoint' in response_data:
                res_queue.put((response_data, retryCount, 'retryEndpoint'))
        elif redis_enabled and retryMethod == 'retryRedis':
            try:
                data = json.dumps(response_data['json'], default=to_serializable) if 'json' in response_data else response_data['data']
                red.set(response_data['redisKey'], data, ex=600)
                if 'deleteRedisKey' in response_data:
                  red.delete(response_data['deleteRedisKey'])

                logger.info('Sending redis response to %s ( %s of %s ) successful',
                    response_data['redisKey'],
                    max_retries - retryCount + 1,
                    max_retries,
                    extra=response_data['log_extras'] if 'log_extras' in response_data else None)
            except Exception as e:
                logger.error('%s', e, extra=response_data['log_extras'] if 'log_extras' in response_data else None)
                if retryCount > 1:
                    logger.info('Sending redis response to %s ( %s of %s ) failed, trying again in %s seconds',
                        response_data['redisKey'],
                        max_retries - retryCount + 1,
                        max_retries,
                        retry_delay_seconds)
                    time.sleep(retry_delay_seconds)
                    res_queue.put((response_data, retryCount - 1, retryMethod))
                else:
                    logger.error('Sending redis response to %s ( %s of %s ) failed finally, no retries anymore',
                        response_data['redisKey'],
                        max_retries - retryCount + 1,
                        max_retries)
        elif retryMethod == 'retryEndpoint':
            boxEndpoint = response_data['boxEndpoint']
            # for testing purposes on local environment
            if 'COACH_DEV_BOX_ENDPOINT' in os.environ:
                boxEndpoint = os.environ.get('COACH_DEV_BOX_ENDPOINT')
            try:
                headers = response_data["header"] if 'header' in response_data else None
                data = json.dumps(response_data['json'], default=to_serializable) if 'json' in response_data else response_data['data']
                res = requests.post(boxEndpoint, headers=headers, data=data)
                if res.status_code != 200:
                    raise Exception('Wrong status code ' + str(res.status_code))

                logger.info('Sending HTTP response to %s ( %s of %s ) successful',
                    boxEndpoint,
                    max_retries - retryCount + 1,
                    max_retries,
                    extra=response_data['log_extras'] if 'log_extras' in response_data else None)
            except Exception as e:
                logger.error('%s', e, extra=response_data['log_extras'] if 'log_extras' in response_data else None)
                if retryCount > 1:
                    logger.info('Sending HTTP response to %s ( %s of %s ) failed, trying again in %s seconds',
                        boxEndpoint,
                        max_retries - retryCount + 1,
                        max_retries,
                        retry_delay_seconds)
                    time.sleep(retry_delay_seconds)
                    res_queue.put((response_data, retryCount - 1, retryMethod))
                else:
                    logger.error('Sending HTTP response to %s ( %s of %s ) failed finally, no retries anymore',
                        boxEndpoint,
                        max_retries - retryCount + 1,
                        max_retries)

def process_requests_worker(req_queue, res_queue, err_queue, processId):
    worker_name = 'process_requests_worker-' + str(processId)
    logger = getLogger(worker_name)
    logger.info(f'Initialize process_requests_worker {worker_name}...')
    calc_count = 0
    while calc_count < maxCalcCount:
        request_data, method = req_queue.get()

        if method == 'calculate_chi2' or method == 'calculate_embeddings':
            logger.error(f'run worker method for {worker_name}.{method}')
            calculate_embeddings_worker(getLogger(f'{worker_name}.calculate_embeddings'), worker_name, req_queue, res_queue, err_queue, request_data, method)
        elif method == 'upload_factcheck_documents':
            logger.error(f'run worker method for {worker_name}.{method}')
            upload_factcheck_documents_worker(getLogger(f'{worker_name}.calculate_embeddings'), worker_name, req_queue, res_queue, err_queue, request_data)
        elif method == 'create_sample_queries':
            logger.error(f'run worker method for {worker_name}.{method}')
            create_sample_queries_worker(getLogger(f'{worker_name}.calculate_embeddings'), worker_name, req_queue, res_queue, err_queue, request_data)
        else:
            logger.error(f'No worker method for {worker_name}.{method}, ignoring.')

        calc_count += 1

def process_requests(req_queue, res_queue, err_queue):
    logger = getLogger('process_requests')
    logger.info('Worker process_requests started...')
    processes = []
    for i in range(int(os.environ.get('COACH_PARALLEL_WORKERS', 1))):
        p = mp.Process(target=process_requests_worker, name=f'process_requests_worker-{i}', args=(req_queue, res_queue, err_queue, i))
        p.daemon = False
        p.start()
        processes.append(p)
    while True:
        for i in range(len(processes)):
            p = processes[i]
            if not p.is_alive():
                p = mp.Process(target=process_requests_worker, name=f'process_requests_worker-{i}', args=(req_queue, res_queue, err_queue, i))
                p.daemon = False
                p.start()
                processes[i] = p

req_queue = mp.Queue()
res_queue = mp.Queue()
err_queue = mp.Queue()

preq = mp.Process(target=process_requests, name='process_requests', args=(req_queue, res_queue, err_queue))
preq.start()
pres = mp.Process(target=process_responses, name='process_responses', args=(req_queue, res_queue, err_queue))
pres.start()
if int(os.environ.get('REDIS_ENABLE', 0)) == 1:
    p = mp.Process(target=process_redis,  name='process_redis', args=(req_queue, res_queue, err_queue))
    p.start()

def create_app():
    app = connexion.App(__name__, specification_dir='openapi/')
    app.add_api('botium_coach_worker_api.yaml')
    app.app.register_blueprint(healthz, url_prefix="/healthz")
    app.app.config.update(
        HEALTHZ={
            "live": "api.health.liveness",
            "ready": "api.health.readiness",
        }
    )
    with app.app.app_context():
        current_app.req_queue = req_queue
        current_app.res_queue = res_queue
        current_app.err_queue = err_queue

    return app

if os.environ.get('GUNICORN_MODE', 0) == 0:
    if __name__ == '__main__':
        print(f'Swagger-UI available at http://127.0.0.1:{os.environ.get("PORT", 4444)}/1.0/ui')
        app = create_app()
        app.run(port=int(os.environ.get('PORT', 4444)))
else:
    app = create_app()

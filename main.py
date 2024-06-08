#!/usr/bin/env python3
import os
import connexion
from functools import singledispatch
from flask import current_app
from flask_healthz import healthz
import multiprocessing as mp
from api.embeddings import calculate_embeddings_worker
from api.factcheck import upload_factcheck_documents_worker, create_sample_queries_worker
import json
import time
import requests
import numpy as np
from api.utils.log import getLogger
import psutil

def killtree(pid, including_parent=True):
    parent = psutil.Process(pid)
    for child in parent.children():
        print('Killing child process %s' % child.pid)
        child.kill()

    if including_parent:
        parent.kill()

max_retries = int(os.environ.get('COACH_RETRY_REQUEST_RETRIES', 12))
retry_delay_seconds = int(os.environ.get('COACH_RETRY_REQUEST_DELAY', 10))
maxCalcCount = int(os.environ.get('COACH_MAX_CALCULATIONS_PER_WORKER', 100))

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

    logger.info('Worker process_responses started...')
    while True:
        response_data, retryCount, retryMethod = (lambda a, b=None, c=None: (a, b, c))(*res_queue.get())

        if retryCount is None:
            retryCount = max_retries

        if retryMethod is None:
            if 'boxEndpoint' in response_data:
                res_queue.put((response_data, retryCount, 'retryEndpoint'))
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

def process_requests_worker(req_queue, res_queue, err_queue, running_queue, processId):
    #os.setpgrp()
    pid = os.getpid()
    worker_name = 'process_requests_worker-' + str(pid) + '-' + str(processId)
    logger = getLogger(worker_name)
    logger.info(f'Initialize process_requests_worker {worker_name}...')

    embeddingsLogger = getLogger(f'{worker_name}.calculate_embeddings')
    fcUploadLogger = getLogger(f'{worker_name}.upload_factcheck_documents')
    fcSampleLogger = getLogger(f'{worker_name}.create_sample_queries')

    calc_count = 0
    while calc_count < maxCalcCount:
        request_data, method = req_queue.get()

        if method == 'calculate_chi2' or method == 'calculate_embeddings':
            logger.info(f'run worker method for {worker_name}.{method}')
            calculate_embeddings_worker(embeddingsLogger, worker_name, req_queue, res_queue, err_queue, running_queue, request_data, method)
        elif method == 'upload_factcheck_documents':
            logger.info(f'run worker method for {worker_name}.{method}')
            upload_factcheck_documents_worker(fcUploadLogger, worker_name, req_queue, res_queue, err_queue, request_data)
        elif method == 'create_sample_queries':
            logger.info(f'run worker method for {worker_name}.{method}')
            create_sample_queries_worker(fcSampleLogger, worker_name, req_queue, res_queue, err_queue, request_data)
        else:
            logger.error(f'No worker method for {worker_name}.{method}, ignoring.')

        calc_count += 1

def process_requests(req_queue, res_queue, err_queue, running_queue, kill_queue):
    pid = os.getpid()
    logger = getLogger('process_requests')
    logger.info('Worker process_requests started...')
    processes = []
    for i in range(int(os.environ.get('COACH_PARALLEL_WORKERS', 1))):
        p = mp.Process(target=process_requests_worker, name=f'process_requests_worker-{str(pid)}-{i}', args=(req_queue, res_queue, err_queue, running_queue, i))
        p.daemon = False
        p.start()
        processes.append(p)
    while True:
        kill_queue.put(None)
        for _pid in iter(kill_queue.get, None):
            logger.info('Killing worker %s', _pid)
            os.kill(_pid, 9)
        for i in range(len(processes)):
            p = processes[i]
            if not p.is_alive():
                p = mp.Process(target=process_requests_worker, name=f'process_requests_worker-{str(pid)}-{i}', args=(req_queue, res_queue, err_queue, running_queue, i))
                p.daemon = False
                p.start()
                processes[i] = p

def process_cancel_worker(req_queue, running_queue, cancel_queue, kill_queue):
    logger = getLogger('process_cancel_worker')
    logger.info('Worker process_cancel_worker started...')
    while True:
        cancel_data = cancel_queue.get()
        testSetId = cancel_data['testSetId']
        logger.info('Killing job for testSetId %s', testSetId)
        while True:
            try:
                running_job = running_queue.get(timeout=5)
                job_data, pid, child_pid = running_job
                logger.info('Checking running job for testSetId %s', job_data['testSetId'])
                if job_data['testSetId'] == testSetId:
                    try:
                        logger.info('Killing worker %s / %s for testSetId %s', pid, os.getpgid(pid), testSetId)
                        kill_queue.put(pid)
                        kill_queue.put(child_pid)
                        logger.info('Killed worker %s for testSetId %s', pid, testSetId)
                    except Exception as e:
                        logger.error('Error killing worker %s for testSetId %s: %s', pid, testSetId, e)
                else:
                    running_queue.put((job_data, pid))
            except Exception as e:
                logger.error('Error checking running jobs: %s', e)
                break
            time.sleep(0.1)
        time.sleep(0.1)
        

req_queue = mp.Queue()
res_queue = mp.Queue()
err_queue = mp.Queue()
running_queue = mp.Queue()
cancel_queue = mp.Queue()
kill_queue = mp.Queue()

preq = mp.Process(target=process_requests, name='process_requests', args=(req_queue, res_queue, err_queue, running_queue, kill_queue))
preq.start()
pres = mp.Process(target=process_responses, name='process_responses', args=(req_queue, res_queue, err_queue))
pres.start()
pcancel = mp.Process(target=process_cancel_worker, name='process_cancel_worker', args=(req_queue, running_queue, cancel_queue, kill_queue))
pcancel.start()

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
        current_app.cancel_queue = cancel_queue

    return app

if os.environ.get('GUNICORN_MODE', 0) == 0:
    if __name__ == '__main__':
        print(f'Swagger-UI available at http://127.0.0.1:{os.environ.get("PORT", 4444)}/1.0/ui')
        app = create_app()
        app.run(port=int(os.environ.get('PORT', 4444)))
else:
    app = create_app()

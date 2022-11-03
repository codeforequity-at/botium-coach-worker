import os
import redis
from flask_healthz import HealthError
from .redis_client import getRedis
from flask import current_app


def check_redis():
    if bool(os.environ.get('REDIS_ENABLE', 0)) == True:
        try:
            red = getRedis(timeout=0.5)
            red.ping()
        except Exception as e:
            raise HealthError('Redis error: {}'.format(str(e)))


def liveness():
    check_redis()
    with current_app.app_context():
        err_queue = current_app.err_queue
        item = None
        try:
            item = err_queue.get_nowait()
            err_queue.put(item)
        except Exception as e:
            None
        if item is not None:
            raise HealthError('Coach worker error: {}'.format(str(item)))


def readiness():
    check_redis()

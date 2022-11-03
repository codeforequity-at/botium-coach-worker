import os
import redis
from flask_healthz import HealthError


def check_redis():
    if bool(os.environ.get('REDIS_ENABLE', 0)) == True:
        try:
            red = redis.Redis(host=os.environ.get('REDIS_HOST', 'localhost'), port=int(os.environ.get(
                'REDIS_PORT', 6379)), db=int(os.environ.get('REDIS_DB', 0)), socket_connect_timeout=1)
            red.ping()
        except Exception as e:
            raise HealthError('Redis error: {}'.format(str(e)))


def liveness():
    check_redis()


def readiness():
    check_redis()

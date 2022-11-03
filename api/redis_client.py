import os
import redis


def getRedis(timeout=600):
    redis_host = os.environ.get('REDIS_HOST', 'localhost')
    redis_port = int(os.environ.get('REDIS_PORT', 6379))
    redis_db = int(os.environ.get('REDIS_DB', 0))
    redis_password = os.environ.get('REDIS_PASSWORD', None)
    redis_sentinel_host = os.environ.get('REDIS_SENTINEL_HOST', None)
    redis_sentinel_port = int(os.environ.get('REDIS_SENTINEL_PORT', 26379))
    redis_sentinel_name = os.environ.get('REDIS_SENTINEL_NAME', None)
    if redis_sentinel_host is not None:
        sentinel = redis.Sentinel(sentinels=[(redis_sentinel_host, redis_sentinel_port)],
                                  db=redis_db, password=redis_password, socket_timeout=timeout)
        return sentinel.master_for(redis_sentinel_name, socket_timeout=timeout)
    else:
        return redis.Redis(host=redis_host,
                           port=redis_port, db=redis_db, password=redis_password, socket_timeout=timeout)

import os
from flask_healthz import HealthError
from flask import current_app

def liveness():
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
    return None
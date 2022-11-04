import logging
from pythonjsonlogger import jsonlogger
import os
from datetime import datetime

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        if not log_record.get('timestamp'):
            # this doesn't use record.created, so it is slightly off
            now = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            log_record['timestamp'] = now
        if log_record.get('level'):
            log_record['level'] = log_record['level'].upper()
        else:
            log_record['level'] = record.levelname
        if log_record.get('name'):
            log_record['name'] = log_record['name'].upper()
        else:
            log_record['name'] = record.name

class ExtraFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        default_attrs = logging.LogRecord(None, None, None, None, None, None, None).__dict__.keys()
        extras = set(record.__dict__.keys()) - default_attrs

        log_items = ['%(levelname)s %(name)s: %(message)s']
        for attr in extras:
            log_items.append(f'{attr}: %({attr})s')
        format_str = f'{", ".join(log_items)}'
        self._style._fmt = format_str

        return super().format(record)

def getLogger(name):
    log_datefmt = '%Y-%m-%d %H:%M:%S'
    log_level = os.environ.get('LOGLEVEL', 'INFO').upper()
    #log_format = '%(asctime)-15s %(clientId)s %(message)s'
    #logging.basicConfig(level=log_level, datefmt=log_datefmt)
    logger = logging.getLogger(name)
    logger.propagate = False
    logHandler = logging.StreamHandler()
    if bool(os.environ.get('JSON_LOGGING', False)) == True:
        formatter = CustomJsonFormatter()
    else:
        formatter = ExtraFormatter()
    logHandler.setFormatter(formatter)
    logger.addHandler(logHandler)
    logger.setLevel(log_level)
    return logger
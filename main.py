#!/usr/bin/env python3
import os
import connexion
import logging

LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(format='%(asctime)-15s %(message)s', level=LOGLEVEL, datefmt='%Y-%m-%d %H:%M:%S')
app = connexion.App(__name__, specification_dir='openapi/')
app.add_api('botium_coach_worker_api.yaml')

if __name__ == '__main__':
  app.run(port=4002)
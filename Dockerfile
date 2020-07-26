FROM tiangolo/meinheld-gunicorn-flask:python3.7

COPY ./Requirements.txt /app/Requirements.txt
RUN pip install -r /app/Requirements.txt
COPY ./setup /app/setup
RUN python /app/setup/download_models.py

COPY ./main.py /app/main.py
COPY ./api /app/api
COPY ./openapi /app/openapi
RUN rm /app/prestart.sh

ENV LOGLEVEL INFO
ENV WEB_CONCURRENCY 1
ENV GUNICORN_CMD_ARGS --timeout 1800 --worker-class gthread

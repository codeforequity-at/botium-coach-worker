FROM tiangolo/meinheld-gunicorn-flask:python3.7

COPY ./Requirements.txt /app/Requirements.txt
RUN pip install -r /app/Requirements.txt
COPY ./setup /app/setup

ENV TFHUB_CACHE_DIR /app/tfhub_modules
ENV NLTK_DATA /app/nltk_data
ENV TRANSFORMERS_CACHE /app/transformers_data
RUN python /app/setup/download_models.py
RUN python -m nltk.downloader -d /app/nltk_data punkt
RUN python -m nltk.downloader -d /app/nltk_data stopwords

COPY ./main.py /app/main.py
COPY ./api /app/api
COPY ./openapi /app/openapi
RUN rm /app/prestart.sh

RUN groupadd -r -g 1000 coach && useradd -r -u 1000 -g 1000 -d /app -s /bin/bash coach
RUN chown -R 1000:1000 /app

ENV LOGLEVEL INFO
ENV WEB_CONCURRENCY 1
ENV COACH_MAX_UTTERANCES_FOR_EMBEDDINGS 500
ENV PORT 8080
ENV GUNICORN_CMD_ARGS --timeout 18000 --worker-class gthread -u coach -g coach

TAG_COMMIT := $(shell git rev-list --abbrev-commit --tags --max-count=1)
COACH_WORKER_VERSION := $(shell git describe --abbrev=0 --tags ${TAG_COMMIT} 2>/dev/null || true)

start_dev:
	COACH_MAX_UTTERANCES_FOR_EMBEDDINGS=500 LOGLEVEL=INFO NLTK_DATA=./nltk_data python3 main.py

docker_build:
	docker build -t botium/botium-coach-worker:$(COACH_WORKER_VERSION) ./

docker_build_dev:
	docker build -t botium/botium-coach-worker:develop ./

docker_run:
	docker run --rm -p 4002:8080 --name botium-coach-worker botium/botium-coach-worker:$(COACH_WORKER_VERSION)

docker_run_dev:
	docker run --rm -p 4002:8080 --name botium-coach-worker botium/botium-coach-worker:develop

docker_bash:
	docker exec -it botium-coach-worker bash

docker_publish:
	docker push botium/botium-coach-worker:$(COACH_WORKER_VERSION)

docker_publish_dev:
	docker push botium/botium-coach-worker:develop

docker_latest:
	docker tag botium/botium-coach-worker:$(COACH_WORKER_VERSION) botium/botium-coach-worker:latest
	docker push botium/botium-coach-worker:latest

install:
	pip3 install -r Requirements.txt

freeze:
	pip3 freeze > Requirements.txt
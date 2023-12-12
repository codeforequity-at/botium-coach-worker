TAG_COMMIT := $(shell git rev-list --abbrev-commit --tags --max-count=1)
COACH_WORKER_VERSION := $(shell git describe --abbrev=0 --tags ${TAG_COMMIT} 2>/dev/null || true)

install:
	pip3 install -r Requirements.txt
	make postinstall

postinstall:
	python ./setup/download_models.py
	python -m nltk.downloader -d ./nltk_data punkt
	python -m nltk.downloader -d ./nltk_data stopwords

freeze:
	pip3 freeze > Requirements.txt

start_dev:
	dotenv -e .env dotenv -e .env.local python3 main.py

docker_build:
	docker build -t botium/botium-coach-worker:$(COACH_WORKER_VERSION) ./

docker_build_dev:
	docker build -t botium/botium-coach-worker:develop ./

docker_run:
	docker run --rm -p 4002:8080 --name botium-coach-worker botium/botium-coach-worker:$(COACH_WORKER_VERSION)

docker_run_dev:
	docker run --rm -p 4002:8080 --name botium-coach-worker botium/botium-coach-worker:develop

docker_run_dev_test:
	docker run --entrypoint '/bin/bash' --rm --name botium-coach-worker botium/botium-coach-worker:develop -c "python3 -m unittest test_python/tests.py"

docker_bash:
	docker exec -it botium-coach-worker bash

docker_publish:
	docker push botium/botium-coach-worker:$(COACH_WORKER_VERSION)

docker_publish_dev:
	docker push botium/botium-coach-worker:develop

docker_latest:
	docker tag botium/botium-coach-worker:$(COACH_WORKER_VERSION) botium/botium-coach-worker:latest
	docker push botium/botium-coach-worker:latest

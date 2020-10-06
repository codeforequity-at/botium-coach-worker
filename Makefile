start_dev:
	LOGLEVEL=INFO python main.py

docker_build:
	docker build -t botium/botium-coach-worker:latest ./

docker_run:
	docker run -p 4002:80 botium/botium-coach-worker:latest

docker_publish:
	docker push botium/botium-coach-worker:latest

install:
	pip3 install -r Requirements.txt

freeze:
	pip3 freeze > Requirements.txt
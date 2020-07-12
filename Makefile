start_dev:
	LOGLEVEL=DEBUG FLASK_APP=main.py FLASK_ENV=development flask run

docker_build:
	docker build -t botium/botium-coach-worker:latest ./

docker_run:
	docker run -p 80:80 botium/botium-coach-worker:latest

docker_publish:
	docker push botium/botium-coach-worker:latest

freeze:
		pip3 freeze > Requirements.txt
start_dev:
	FLASK_APP=app.py FLASK_ENV=development flask run

freeze:
		pip3 freeze > Requirements.txt
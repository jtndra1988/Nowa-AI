.PHONY: build up down logs test migrate

build:
	docker-compose build

up:
	docker-compose up

down:
	docker-compose down

logs:
	docker-compose logs -f

test:
	docker-compose exec backend pytest

migrate:
	docker-compose exec backend alembic upgrade head

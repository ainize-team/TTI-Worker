# TTI-Worker

Serving Text To Image Model Using FastAPI and Celery

## For Developers

1. install dev package.

```shell
poetry install
```

2. install pre-commit.

```shell
pre-commit install
```

## Running the Celery worker server with RabbitMQ(as Broker), Redis(as Backend).
### Running RabbitMQ server using Docker
```shell
docker run -d --name rabbitmq -p 5672:5672 -p 8080:15672 --restart=unless-stopped rabbitmq:3.9.21-management
```

### Running Redis server using Docker
```
docker run --name redis -d -p 6379:6379 redis
```
- Want to control Redis
  - `sudo apt install redis-tools` in local shell


## 3-1. Using docker
### Build docker file
```
git clone https://github.com/ainize-team/TTI-Worker.git
cd TTI-Worker
docker build -t celery-tti .
```

### Run docker container
```
docker run -d --name <worker_container_name>
--gpus='"device=0"'
-e BROKER_URI=<broker_uri> -e REDIS_HOST=<redis_hostname> 
-e REDIS_PORT=<redis_port> -e REDIS_DB=<redis_db> 
-e REDIS_PASSWORD=<redis_password> -v <local-path>:/model 
celery-tti
```

### Test with FastAPI
- Check our [TTI-FastAPI](https://github.com/ainize-team/TTI-FastAPI) Repo.

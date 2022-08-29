# TTI-Worker

Serving Text To Image Model Using FastAPI and Celery

## For Developers

1. install dev package.

```shell
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

2. install pre-commit.

```shell
pre-commit install
```

## Running the Celery worker server with RabbitMQ(as Broker) Firebase(as Backend).
### Running RabbitMQ server using Docker
```shell
docker run -d --name rabbitmq -p 5672:5672 -p 8080:15672 --restart=unless-stopped rabbitmq:3.9.21-management
```

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
--gpus='"device=0"' -e BROKER_URI=<broker_uri> 
-e DATABASE_URL=<firebase_realtime_database_url> 
-e STORAGE_BUCKET=<firebase_storage_url>  -v <firebase_credential_dir_path>:/app/key-v <local-path>:/app/model 
celery-tti
```

### Test with FastAPI
- Check our [TTI-FastAPI](https://github.com/ainize-team/TTI-FastAPI) Repo.

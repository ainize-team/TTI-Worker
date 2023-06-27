# TTI-Worker

Serving Text To Image Model Using FastAPI and Celery



## How to start
### Using docker-compose(recommended)
1. Create RabbitMQ and FastAPI container(refer to [TTI-FastAPI](https://github.com/ainize-team/TTI-FastAPI))

2. Clone repository
```shell
git clone https://github.com/ainize-team/TTI-Worker
cd TTI-Worker
```

3. Edit [docker-compose.yml](./docker-compose.yml) and [.env file](./.env.sample) for your project.
- Vhost for each worker is in [docker-compose.yml](./docker-compose.yml) and common RabbitMQ config is in [.env file](./.env.sample).

4. Run worker container
```shell
docker-compose up -d

# If you want to run a specific worker container, write service name.
docker-compose up -d <service name>
```

### Using docker
1. Run RabbitMQ comtainer as a broker
```shell
docker run -d --name tti-rabbitmq -p 5672:5672 -p 15672:15672 --restart=unless-stopped rabbitmq:3.11.2-management
```

2. Clone repository
```shell
git clone https://github.com/ainize-team/TTI-Worker
cd TTI-Worker
```

3. Build docker image
```shell
docker build -t tti-worker .
```

4. Create docker container
```shell
docker run -d --name <worker_container_name> \
--gpus='"device=0"' -e BROKER_URI=<broker_uri> \
-e DATABASE_URL=<firebase_realtime_database_url> \
-e STORAGE_BUCKET=<firebase_storage_url> \
-v <firebase_credential_path>:/app/key -v <model_local_path>:/app/model \
tti-worker
```
Or, you can use the [.env file](./.env.sample) to run as follows.
```shell
docker run -d --name <worker_container_name> \
--gpus='"device=0"' \
--env-file <env filename> \
-v <firebase_credential_path>:/app/key -v <model_local_path>:/app/model \
tti-worker
```

## Test with FastAPI
- Check our [TTI-FastAPI](https://github.com/ainize-team/TTI-FastAPI) Repo.


## For Developers

1. Install dev package

```shell
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

2. Install pre-commit

```shell
pre-commit install
```
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
docker run -d --name {worker_container_name} 
--gpus='"device=4,5,6,7"'
-e BROKER_URI={broker_uri} -e REDIS_HOST=<redis_hostname> 
-e REDIS_PORT=<redis_port> -e REDIS_DB=<redis_db> 
-e REDIS_PASSWORD=<redis_password> -v {local-path}:/model 
celery-tti
```

### Test with FastAPI
- Check our [TTI-FastAPI](https://github.com/ainize-team/TTI-FastAPI) Repo.


### Test with local file
```python
import os
from celery.result import AsyncResult
from tasks import generate

os.environ["broker_uri"] = "amqp://guest:guest@localhost:8080//"


task = generate.delay("random_task_id", {
  "prompt": "deer shaped lamp",
  "outdir": "latent-diffusion/outputs",
  "ddim_steps": 30,
  "ddim_eta": 0,
  "n_iter": 1,
  "W": 256,
  "H": 256,
  "n_samples": 2,
  "scale": 5,
  "plms": True,
})

# Get task result
print(task.get())
```
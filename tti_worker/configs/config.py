from pydantic import BaseSettings, Field


class CeleryWorkerSettings(BaseSettings):
    worker_name: str = "Celery Worker"
    broker_uri: str
    backend_uri: str = "Auto Generate"


class ModelSettings(BaseSettings):
    model_path: str = "./model"


class RedisSettings(BaseSettings):
    redis_host: str = "localhost"
    redis_port: int = Field(
        default=6379,
        ge=0,
        le=65535,
    )
    redis_db: int = Field(
        default=0,
        ge=0,
        le=15,
    )
    redis_password: str = ""


celery_worker_settings = CeleryWorkerSettings()
model_settings = ModelSettings()
redis_settings = RedisSettings()

celery_worker_settings.backend_uri = "redis://:{password}@{hostname}:{port}/{db}".format(
    hostname=redis_settings.redis_host,
    password=redis_settings.redis_password,
    port=redis_settings.redis_port,
    db=redis_settings.redis_db,
)

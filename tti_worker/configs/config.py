from pydantic import BaseSettings, Field


class CeleryWorkerSettings(BaseSettings):
    worker_name: str = "Celery Worker"
    broker_uri: str = "amqp://guest:guest@localhost:5672//"
    backend_uri: str = "Auto Generate"


class ModelSettings(BaseSettings):
    model_name_or_path: str = "./model"
    model_output_path: str = "./outputs"


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


class FirebaseSettings(BaseSettings):
    cred_path: str
    database_url: str
    storage_bucket: str
    app_name: str = "text-to-image"


celery_worker_settings = CeleryWorkerSettings()
model_settings = ModelSettings()
redis_settings = RedisSettings()
firebase_settings = FirebaseSettings()

celery_worker_settings.backend_uri = "redis://:{password}@{hostname}:{port}/{db}".format(
    hostname=redis_settings.redis_host,
    password=redis_settings.redis_password,
    port=redis_settings.redis_port,
    db=redis_settings.redis_db,
)

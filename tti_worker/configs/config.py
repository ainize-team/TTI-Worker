from pydantic import BaseSettings


class CeleryWorkerSettings(BaseSettings):
    worker_name: str = "Celery Worker"
    broker_uri: str = "amqp://guest:guest@localhost:5672//"
    backend_uri: str = "Auto Generate"


class ModelSettings(BaseSettings):
    model_name_or_path: str = "./model"
    model_output_path: str = "./outputs"


class FirebaseSettings(BaseSettings):
    cred_path: str
    database_url: str
    storage_bucket: str
    app_name: str = "text-to-image"


celery_worker_settings = CeleryWorkerSettings()
model_settings = ModelSettings()
firebase_settings = FirebaseSettings()

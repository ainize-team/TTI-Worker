from pydantic import BaseSettings, Field

from enums import ModelClassNameEnums


class CeleryWorkerSettings(BaseSettings):
    worker_name: str = "Celery Worker"
    broker_uri: str = "amqp://guest:guest@localhost:5672//"


class ModelSettings(BaseSettings):
    model_class_name: ModelClassNameEnums = ModelClassNameEnums.STABLE_DIFFUSION
    model_name_or_path: str = "./model"
    model_output_path: str = "./outputs"
    model_unit_memory: int = Field(default=3072, description="Memory Required to Create an Image 512 by 512")


class FirebaseSettings(BaseSettings):
    cred_path: str = "/app/key/serviceAccountKey.json"
    database_url: str
    storage_bucket: str
    app_name: str = "text-to-art"


celery_worker_settings = CeleryWorkerSettings()
model_settings = ModelSettings()
firebase_settings = FirebaseSettings()

import firebase_admin
from celery import Celery
from configs import celery_config
from configs.config import celery_worker_settings, firebase_settings, redis_settings
from firebase_admin import credentials
from redis import Redis


app = Celery(
    celery_worker_settings.worker_name,
    broker=celery_worker_settings.broker_uri,
    backend=celery_worker_settings.backend_uri,
    include=["tasks"],
)
app.config_from_object(celery_config)
redis = Redis(
    host=redis_settings.redis_host,
    port=redis_settings.redis_port,
    db=redis_settings.redis_db,
    password=redis_settings.redis_password,
)

cred = credentials.Certificate(firebase_settings.cred_path)
firebase_admin.initialize_app(
    cred,
    {
        "databaseURL": firebase_settings.database_url,
        "storageBucket": firebase_settings.storage_bucket,
    },
)

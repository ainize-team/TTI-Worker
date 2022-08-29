import firebase_admin
from celery import Celery
from configs import celery_config
from configs.config import celery_worker_settings, firebase_settings
from firebase_admin import credentials


app = Celery(
    celery_worker_settings.worker_name,
    broker=celery_worker_settings.broker_uri,
    include=["tasks"],
)
app.config_from_object(celery_config)

cred = credentials.Certificate(firebase_settings.cred_path)
firebase_admin.initialize_app(
    cred,
    {
        "databaseURL": firebase_settings.database_url,
        "storageBucket": firebase_settings.storage_bucket,
    },
)

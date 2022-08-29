import gc
import os
import shutil
from datetime import datetime

import pytz
import torch
from configs.config import firebase_settings
from firebase_admin import db, storage
from schemas import ImageGenerationRequest, ImageGenerationResponse


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_now_timestamp():
    return datetime.utcnow().replace(tzinfo=pytz.utc).timestamp()


def save_task_data(task_id: str, user_request: ImageGenerationRequest, response: ImageGenerationResponse):
    app_name = firebase_settings.app_name
    task_data = user_request.dict()
    task_data.update(response.dict())
    db.reference(app_name).child(task_id).set(task_data)


def update_response(task_id: str, response: ImageGenerationResponse):
    app_name = firebase_settings.app_name
    db.reference(app_name).child(task_id).update(response.dict())


def upload_output_images(task_id: str, output_path: str):
    app_name = firebase_settings.app_name
    urls = {}
    bucket = storage.bucket()
    for filename in os.listdir(output_path):
        fn = os.path.splitext(filename)[0]
        blob = bucket.blob(f"{app_name}/results/{task_id}/{filename}")
        blob.upload_from_filename(os.path.join(output_path, filename))
        blob.make_public()
        url = blob.public_url
        urls[fn] = url

    return urls


def remove_output_images(output_path: str):
    shutil.rmtree(output_path)

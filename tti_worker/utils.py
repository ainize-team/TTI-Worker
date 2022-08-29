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
    task_data = dict(user_request)
    task_data.update(dict(response))
    db.reference(app_name).child(task_id).set(dict(task_data))


def update_response(task_id: str, response: ImageGenerationResponse):
    if response.error is not None:
        response.error = dict(response.error)
    app_name = firebase_settings.app_name
    db.reference(app_name).child(task_id).update(dict(response))


def upload_output_images(task_id: str, output_path: str):
    app_name = firebase_settings.app_name
    paths = {}
    bucket = storage.bucket()
    for filename in os.listdir(output_path):
        fn = os.path.splitext(filename)[0]
        blob = bucket.blob(f"{app_name}/results/{task_id}/{filename}")
        blob.upload_from_filename(os.path.join(output_path, filename))
        blob.make_public()
        path = blob.public_url
        paths[fn] = path

    return paths


def remove_output_images(output_path: str):
    shutil.rmtree(output_path)

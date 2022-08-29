import gc
import os
import shutil
from datetime import datetime

import pytz
import torch
from configs.config import firebase_settings
from firebase_admin import db, storage
from schemas import ImageGenerationErrorResponse, ImageGenerationRequest, ImageGenerationResponse


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
    app_name = firebase_settings.app_name
    db.reference(app_name).child(task_id).update(dict(response))


def update_error_message(task_id: str, error_response: ImageGenerationErrorResponse):
    app_name = firebase_settings.app_name
    db.reference(app_name).child(task_id).update(dict(error_response))


def save_output_images(task_id: str, output_path: str):
    app_name = firebase_settings.app_name
    bucket = storage.bucket()
    for filename in os.listdir(output_path):
        blob = bucket.blob(f"{app_name}/results/{task_id}/{filename}")
        blob.upload_from_filename(filename)


def remove_output_images(output_path: str):
    shutil.rmtree(output_path)

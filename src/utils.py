import gc
import os
from datetime import datetime
from typing import Dict, List

import pytz
import torch
from firebase_admin import db, storage

from configs.config import firebase_settings
from schemas import ImageGenerationRequest, ImageGenerationResponse, ImageGenerationResult, ImageGenerationWorkerOutput


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


def upload_output_images(task_id: str, results: List[ImageGenerationWorkerOutput]) -> Dict[str, ImageGenerationResult]:
    app_name = firebase_settings.app_name
    ret: Dict[str, ImageGenerationResult] = {}
    bucket = storage.bucket()
    for result in results:
        image_path = result.image_path
        base_name = os.path.basename(image_path)
        file_name = os.path.splitext(base_name)[0]

        blob = bucket.blob(f"{app_name}/results/{task_id}/{base_name}")
        blob.upload_from_filename(image_path)
        blob.make_public()

        url = blob.public_url

        ret[file_name] = ImageGenerationResult(
            url=url, is_filtered=result.nsfw_content_detected, base_seed=result.base_seed, image_no=result.image_no
        )
        os.remove(image_path)

        if result.origin_image_path is not None:
            origin_image_path = result.origin_image_path
            origin_base_name = os.path.basename(origin_image_path)

            origin_blob = bucket.blob(f"{app_name}/results/{task_id}/{origin_base_name}")
            origin_blob.upload_from_filename(origin_image_path)
            origin_blob.make_public()
            origin_url = origin_blob.public_url
            ret[file_name].origin_url = origin_url
            os.remove(origin_image_path)

    return ret

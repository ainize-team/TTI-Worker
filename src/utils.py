import gc
import os
import random
import shutil
import string
from datetime import datetime
from typing import Dict, List

import torch
from firebase_admin import db, storage

from configs.config import firebase_settings, model_settings
from schemas import ImageGenerationResponse, ImageGenerationResult, ImageGenerationWorkerOutput


app_name = firebase_settings.app_name


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_random_string(n: int = 16) -> str:
    chars = string.ascii_letters + string.digits
    return "".join(random.choice(chars) for _ in range(n))


def get_now_timestamp() -> int:
    return int(datetime.utcnow().timestamp() * 1000)


def update_response(task_id: str, response: ImageGenerationResponse):
    db.reference(f"{app_name}/tasks/{task_id}").update(response.dict())


def upload_output_images(task_id: str, results: List[ImageGenerationWorkerOutput]) -> Dict[str, ImageGenerationResult]:
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

        if result.origin_image_path is not None:
            origin_image_path = result.origin_image_path
            origin_base_name = os.path.basename(origin_image_path)

            origin_blob = bucket.blob(f"{app_name}/results/{task_id}/{origin_base_name}")
            origin_blob.upload_from_filename(origin_image_path)
            origin_blob.make_public()
            ret[file_name].origin_url = origin_blob.public_url

    shutil.rmtree(os.path.join(model_settings.model_output_path, task_id), ignore_errors=True)

    return ret

import base64
import json
from datetime import datetime
from functools import partialmethod
from typing import Dict

import pytz
from celery.signals import celeryd_init
from enums import ResponseStatusEnum
from loguru import logger
from ml_model import TextToImageModel
from schemas import ImageGenerationRequest, ImageGenerationResponse
from tqdm import tqdm
from utils import clear_memory
from worker import app, redis


tti = TextToImageModel()


@celeryd_init.connect
def load_model(**kwargs):
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    logger.info("Start loading model...")
    tti.load_model()
    tti.load_clip_model()
    logger.info("Loading model is done!")


@app.task(name="generate")
def generate(task_id: str, data: Dict) -> str:
    now = datetime.utcnow().replace(tzinfo=pytz.utc).timestamp()
    response = ImageGenerationResponse(status=ResponseStatusEnum.ASSIGNED, updated_at=now)
    redis.set(task_id, json.dumps(dict(response)))
    user_request: ImageGenerationRequest = ImageGenerationRequest(
        prompt=data["prompt"],
        steps=data["steps"],
        width=data["width"],
        height=data["height"],
        images=data["images"],
        guidance_scale=data["guidance_scale"],
    )
    try:
        # TODO: remove this code
        grid_image_path = tti.generate(task_id, user_request)
        with open(grid_image_path, "rb") as f:
            data = f.read()
        base64_str = base64.b64encode(data).decode("utf-8")
        now = datetime.utcnow().replace(tzinfo=pytz.utc).timestamp()
        response.updated_at = now
        response.status = ResponseStatusEnum.COMPLETED
        response.result = base64_str
        logger.info(f"task_id: {task_id} is done")
        redis.set(task_id, json.dumps(dict(response)))
    except ValueError as e:
        redis.set(task_id, json.dumps({"status_code": 422, "message": str(e)}))
        return str(e)
    except Exception as e:
        redis.set(task_id, json.dumps({"status_code": 500, "message": str(e)}))
        return str(e)
    finally:
        clear_memory()

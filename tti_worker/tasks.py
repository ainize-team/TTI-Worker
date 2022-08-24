import argparse
import json
from datetime import datetime
from typing import Dict

import pytz
from celery.signals import celeryd_init
from enums import ResponseStatusEnum
from loguru import logger
from ml_model import TextToImageModel
from payloads.response import ImageGenerationResponse
from utils import clear_memory
from worker import app, redis


tti = TextToImageModel()


@celeryd_init.connect
def load_model(**kwargs):
    logger.info("Start loading model...")
    tti.load_model()
    tti.load_clip_model()

    logger.info("Loading model is done!")


@app.task(name="generate")
def generate(task_id: str, data: Dict) -> str:
    print("generate start")
    now = datetime.utcnow().replace(tzinfo=pytz.utc).timestamp()
    response = ImageGenerationResponse(status=ResponseStatusEnum.ASSIGNED, updated_at=now)
    redis.set(task_id, json.dumps(dict(response)))
    opt = argparse.Namespace(
        prompt=data["prompt"],
        ddim_steps=data["ddim_steps"],
        ddim_eta=0,
        n_iter=1,
        W=data["W"],
        H=data["H"],
        n_samples=int(data["n_samples"]),
        scale=data["scale"],
        plms=True,
    )
    error_flag = False
    try:
        response.result = tti.generate(opt, task_id)
    except ValueError as e:
        redis.set(task_id, json.dumps({"status_code": 422, "message": str(e)}))
        error_flag = True
        return str(e)
    except Exception as e:
        redis.set(task_id, json.dumps({"status_code": 500, "message": str(e)}))
        error_flag = True
        return str(e)
    finally:
        clear_memory()
    if not error_flag:
        now = datetime.utcnow().replace(tzinfo=pytz.utc).timestamp()
        response.updated_at = now
        response.status = ResponseStatusEnum.COMPLETED
        logger.info(f"task_id: {task_id}, gen result: {response.result}")
        redis.set(task_id, json.dumps(dict(response)))
        return "success"

from functools import partialmethod
from typing import Dict

from celery.signals import celeryd_init
from enums import ResponseStatusEnum
from loguru import logger
from ml_model import TextToImageModel
from schemas import ImageGenerationErrorResponse, ImageGenerationRequest, ImageGenerationResponse
from tqdm import tqdm
from utils import (
    clear_memory,
    get_now_timestamp,
    remove_output_images,
    save_output_images,
    save_task_data,
    update_error_message,
    update_response,
)
from worker import app


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
    now = get_now_timestamp()
    response = ImageGenerationResponse(status=ResponseStatusEnum.ASSIGNED, updated_at=now)
    user_request: ImageGenerationRequest = ImageGenerationRequest(**data)
    save_task_data(task_id, user_request, response)
    try:
        output_path = tti.generate(task_id, user_request)
        save_output_images(task_id, output_path)
        now = get_now_timestamp()
        response.updated_at = now
        response.status = ResponseStatusEnum.COMPLETED
        logger.info(f"task_id: {task_id} is done")
        update_response(task_id, response)
        remove_output_images(output_path)
    except ValueError as e:
        error_response = ImageGenerationErrorResponse(
            status_code=422, status=ResponseStatusEnum.ERROR, message=str(e), updated_at=now
        )
        update_error_message(task_id, error_response)
    except Exception as e:
        error_response = ImageGenerationErrorResponse(
            status_code=422, status=ResponseStatusEnum.ERROR, message=str(e), updated_at=now
        )
        update_error_message(task_id, error_response)
    finally:
        clear_memory()

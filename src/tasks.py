from functools import partialmethod
from typing import Dict, List

from celery.signals import celeryd_init
from loguru import logger
from tqdm import tqdm

from enums import ErrorStatusEnum, ResponseStatusEnum
from ml_model import TextToImageModel
from schemas import Error, ImageGenerationRequest, ImageGenerationResponse, ImageGenerationWorkerOutput
from utils import clear_memory, get_now_timestamp, update_response, upload_output_images
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
    response = ImageGenerationResponse(status=ResponseStatusEnum.ASSIGNED, updated_at=get_now_timestamp())
    update_response(task_id, response)
    try:
        user_request: ImageGenerationRequest = ImageGenerationRequest(**data)
        results: List[ImageGenerationWorkerOutput] = tti.generate(task_id, user_request)
        response.response = upload_output_images(task_id, results)
        response.status = ResponseStatusEnum.COMPLETED
        response.updated_at = get_now_timestamp()
        update_response(task_id, response)
        logger.info(f"task_id: {task_id} is done")
    except ValueError as e:
        error = Error(status_code=ErrorStatusEnum.UNPROCESSABLE_ENTITY, error_message=str(e))
        error_response = ImageGenerationResponse(
            status=ResponseStatusEnum.ERROR, error=error, updated_at=get_now_timestamp()
        )
        update_response(task_id, error_response)
    except Exception as e:
        error = Error(status_code=ErrorStatusEnum.INTERNAL_SERVER_ERROR, error_message=str(e))
        error_response = ImageGenerationResponse(
            status=ResponseStatusEnum.ERROR, error=error, updated_at=get_now_timestamp()
        )
        update_response(task_id, error_response)
    finally:
        clear_memory()

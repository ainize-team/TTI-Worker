from functools import partialmethod
from typing import Dict

from celery.signals import celeryd_init
from enums import ErrorStatusEnum, ResponseStatusEnum
from loguru import logger
from ml_model import TextToImageModel
from schemas import Error, ImageGenerationRequest, ImageGenerationResponse
from tqdm import tqdm
from utils import (
    clear_memory,
    get_now_timestamp,
    remove_output_images,
    save_task_data,
    update_response,
    upload_output_images,
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
    response = ImageGenerationResponse(status=ResponseStatusEnum.ASSIGNED, updated_at=get_now_timestamp())
    user_request: ImageGenerationRequest = ImageGenerationRequest(**data)
    save_task_data(task_id, user_request, response)
    try:
        output_path, filter_results = tti.generate(task_id, user_request)
        urls = upload_output_images(task_id, output_path)  # {0: ~~~, 1: ~~~, grid: ~~~}
        response.results = {}
        for i in urls.keys():
            result = {}
            result["url"] = urls[i]
            if i != "grid":
                result["is_filtered"] = filter_results[int(i) - 1]
            response.results[i] = result
        response.status = ResponseStatusEnum.COMPLETED
        response.updated_at = get_now_timestamp()
        update_response(task_id, response)
        remove_output_images(output_path)
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

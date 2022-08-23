from typing import Union

from enums import ResponseStatusEnum
from pydantic import BaseModel


class ImageGenerationResponse(BaseModel):
    status: ResponseStatusEnum = ResponseStatusEnum.PENDING
    updated_at: float = 0.0
    result: Union[str, None] = None

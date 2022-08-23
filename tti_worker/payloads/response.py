from typing import List, Union

from enums import ResponseStatusEnum
from pydantic import BaseModel


class TextGenerationResponse(BaseModel):
    status: ResponseStatusEnum = ResponseStatusEnum.PENDING
    updated_at: float = 0.0
    result: Union[List[str], None] = None

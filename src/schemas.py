import os
from typing import Dict, Optional, Union

from pydantic import BaseModel, Field, HttpUrl

from enums import ResponseStatusEnum


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(
        ...,
        description="try adding increments to your prompt such as 'oil on canvas', 'a painting', 'a book cover'",
    )
    steps: int = Field(
        default=45, ge=1, le=100, description="more steps can increase quality but will take longer to generate"
    )
    seed: int = Field(default=1, ge=0, le=2147483647)
    width: int = Field(default=512, ge=32, le=1024)
    height: int = Field(default=512, ge=32, le=1024)
    images: int = Field(2, ge=1, le=4, description="How many images you wish to generate")
    guidance_scale: float = Field(7.5, ge=0, le=50, description="how much the prompt will influence the results")


class Error(BaseModel):
    status_code: int
    error_message: str


class ImageGenerationWorkerOutput(BaseModel):
    image_path: Union[str, os.PathLike, None] = None
    nsfw_content_detected: bool = False
    base_seed: int = Field(default=1, ge=0, le=2147483647)
    image_no: int = Field(default=0, ge=0, le=4, description="Image number, 0 is grid image")


class ImageGenerationResult(BaseModel):
    url: HttpUrl
    is_filtered: Optional[bool]
    base_seed: int = Field(default=1, ge=0, le=2147483647)
    image_no: int = Field(default=0, ge=0, le=4, description="Image number, 0 is grid image")


class ImageGenerationResponse(BaseModel):
    status: ResponseStatusEnum = ResponseStatusEnum.PENDING
    results: Dict[str, ImageGenerationResult] = None
    error: Union[None, Error] = None
    updated_at: float = 0.0

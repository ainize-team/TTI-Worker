from typing import Dict, Optional, Union

from pydantic import BaseModel, Field, HttpUrl

from enums import ResponseStatusEnum, SchedulerType


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(
        ...,
        description="try adding increments to your prompt such as 'oil on canvas', 'a painting', 'a book cover'",
    )
    negative_prompt: Optional[str] = Field(
        "",
        description="negative prompting indicates which terms you do not want to see in the resulting image.",
    )
    steps: int = Field(
        default=50, ge=1, le=100, description="more steps can increase quality but will take longer to generate"
    )
    seed: int = Field(default=1, ge=0, le=4294967295)
    width: int = Field(default=512, ge=512, le=1024)
    height: int = Field(default=512, ge=512, le=1024)
    images: int = Field(2, ge=1, le=4, description="How many images you wish to generate")
    guidance_scale: float = Field(7.5, ge=0, le=50, description="how much the prompt will influence the results")
    scheduler_type: SchedulerType = Field(SchedulerType.DDIM, description="Scheduler Type")


class Error(BaseModel):
    status_code: int
    error_message: str


class ImageGenerationWorkerOutput(BaseModel):
    image_path: str
    origin_image_path: Optional[str]
    nsfw_content_detected: bool = False
    base_seed: int = Field(default=1, ge=0, le=4294967295)
    image_no: int = Field(default=0, ge=0, le=4, description="Image number, 0 is grid image")


class ImageGenerationResult(BaseModel):
    url: HttpUrl
    origin_url: Optional[HttpUrl]
    is_filtered: bool
    base_seed: int = Field(default=1, ge=0, le=4294967295)
    image_no: int = Field(default=0, ge=0, le=4, description="Image number, 0 is grid image")


class ImageGenerationResponse(BaseModel):
    status: ResponseStatusEnum = ResponseStatusEnum.PENDING
    response: Dict[str, ImageGenerationResult] = None
    error: Union[None, Error] = None
    updated_at: int = 0

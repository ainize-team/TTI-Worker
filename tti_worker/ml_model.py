import os
from typing import List, Union

import torch
from configs.config import model_settings
from diffusers import DiffusionPipeline
from PIL import Image
from schemas import ImageGenerationRequest
from torch import autocast


class TextToImageModel:
    def __init__(self):
        self.clip_model = None
        self.model: Union[DiffusionPipeline, None] = None
        os.makedirs(model_settings.model_output_path, exist_ok=True)
        self.load_model()

    def load_model(self) -> None:
        if torch.cuda.is_available():
            self.diffusion_pipeline = DiffusionPipeline.from_pretrained(
                model_settings.model_name_or_path, torch_dtype=torch.float16
            ).to("cuda")
        else:
            self.diffusion_pipeline = DiffusionPipeline.from_pretrained(model_settings.model_name_or_path)

    def generate(self, task_id: str, data: ImageGenerationRequest) -> str:
        def make_grid(images: List[Image.Image]):
            rows = 1
            cols = len(images)
            if len(images) == 4:
                rows = 2
                cols = 2
            w, h = images[0].size
            grid = Image.new("RGB", size=(cols * w, rows * h))
            for i, img in enumerate(images):
                grid.paste(img, box=(i % cols * w, i // cols * h))
            return grid

        if torch.cuda.is_available():
            generator = torch.cuda.manual_seed(data.seed)
            with autocast("cuda"):
                result = self.diffusion_pipeline(
                    prompt=[data.prompt] * data.images,
                    generator=generator,
                    height=data.height,
                    width=data.width,
                    num_inference_steps=data.steps,
                )
                images: List[Image.Image] = result["sample"]
                filter_results: List[bool] = result["nsfw_content_detected"]
            torch.cuda.empty_cache()
        else:
            generator = torch.manual_seed(data.seed)
            result = self.diffusion_pipeline(
                prompt=[data.prompt] * data.images,
                generator=generator,
                height=data.height,
                width=data.width,
                num_inference_steps=data.steps,
            )
            images: List[Image.Image] = result["sample"]
            filter_results: List[bool] = result["nsfw_content_detected"]

        output_path = os.path.join(model_settings.model_output_path, task_id)
        os.makedirs(output_path, exist_ok=True)

        grid_image: Image.Image = make_grid(images)
        grid_image.save(os.path.join(output_path, "grid.png"))
        for i in range(data.images):
            images[i].save(os.path.join(output_path, f"{i + 1}.png"))

        return output_path, filter_results

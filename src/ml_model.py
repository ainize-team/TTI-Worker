import os
from typing import List, Union

import nvgpu
import torch
from diffusers import DiffusionPipeline
from PIL import Image
from torch import autocast

from configs.config import model_settings
from schemas import ImageGenerationRequest, ImageGenerationWorkerOutput


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

    def generate(self, task_id: str, data: ImageGenerationRequest) -> List[ImageGenerationWorkerOutput]:
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
            gpu_info = nvgpu.gpu_info()[0]
            mem_free = gpu_info["mem_total"] - gpu_info["mem_used"]
            estimated_memory_usage_per_image = (
                model_settings.model_unit_memory * ((data.width / 512) ** 2) * ((data.height / 512) ** 2)
            )
            if estimated_memory_usage_per_image * data.images < mem_free:
                images: List[Image.Image] = []
                filter_results: List[bool] = []
                base_seed_list: List[int] = []
                image_no_list: List[int] = []
                for seed in range(data.seed, data.seed + data.images):
                    seed = seed & 2147483647
                    generator = torch.cuda.manual_seed(seed)
                    with autocast("cuda"):
                        result = self.diffusion_pipeline(
                            prompt=[data.prompt],
                            generator=generator,
                            height=data.height,
                            width=data.width,
                            guidance_scale=data.guidance_scale,
                            num_inference_steps=data.steps,
                        )
                        image: List[Image.Image] = result["sample"][0]
                        if "nsfw_content_detected" in result:
                            filter_result: bool = result["nsfw_content_detected"][0]
                        else:
                            filter_result: bool = False
                        torch.cuda.empty_cache()
                        images.append(image)
                        filter_results.append(filter_result)
                        base_seed_list.append(seed)
                        image_no_list.append(1)
                    torch.cuda.empty_cache()
            else:
                generator = torch.cuda.manual_seed(data.seed)
                with autocast("cuda"):
                    result = self.diffusion_pipeline(
                        prompt=[data.prompt] * data.images,
                        generator=generator,
                        height=data.height,
                        width=data.width,
                        guidance_scale=data.guidance_scale,
                        num_inference_steps=data.steps,
                    )
                    images: List[Image.Image] = result["sample"]
                    if "nsfw_content_detected" in result:
                        filter_results: List[bool] = result["nsfw_content_detected"]
                    else:
                        filter_results: List[bool] = [False] * len(images)
                    base_seed_list: List[int] = [data.seed] * len(images)
                    image_no_list: List[int] = [no for no in range(1, data.images + 1)]
                torch.cuda.empty_cache()
        else:
            generator = torch.manual_seed(data.seed)
            result = self.diffusion_pipeline(
                prompt=[data.prompt] * data.images,
                generator=generator,
                height=data.height,
                width=data.width,
                guidance_scale=data.guidance_scale,
                num_inference_steps=data.steps,
            )
            images: List[Image.Image] = result["sample"]
            if "nsfw_content_detected" in result:
                filter_results: List[bool] = result["nsfw_content_detected"]
            else:
                filter_results: List[bool] = [False] * len(images)
            base_seed_list: List[int] = [data.seed] * len(images)
            image_no_list: List[int] = [no for no in range(1, data.images + 1)]

        output_path = os.path.join(model_settings.model_output_path, task_id)
        os.makedirs(output_path, exist_ok=True)

        result: List[ImageGenerationWorkerOutput] = []

        grid_image: Image.Image = make_grid(images)
        grid_image.save(os.path.join(output_path, "grid.png"))
        result.append(
            ImageGenerationWorkerOutput(
                image_path=os.path.join(output_path, "grid.png"),
                nsfw_content_detected=bool(sum(filter_results)),
                base_seed=data.seed,
                image_no=0,
            )
        )
        for i in range(data.images):
            images[i].save(os.path.join(output_path, f"{i + 1}.png"))
            result.append(
                ImageGenerationWorkerOutput(
                    image_path=os.path.join(output_path, f"{i + 1}.png"),
                    nsfw_content_detected=filter_results[i],
                    base_seed=base_seed_list[i],
                    image_no=image_no_list[i],
                )
            )

        return result

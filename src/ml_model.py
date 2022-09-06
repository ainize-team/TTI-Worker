import os
from asyncio.log import logger
from typing import List, Union

import nvgpu
import torch
import torch.nn as nn
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from PIL import Image
from torch import autocast
from transformers import CLIPConfig, CLIPVisionModel, PreTrainedModel

from configs.config import model_settings
from enums import ModelClassNameEnums
from schemas import ImageGenerationRequest, ImageGenerationWorkerOutput
from utils import clear_memory


def cosine_distance(image_embeds, text_embeds):
    normalized_image_embeds = nn.functional.normalize(image_embeds)
    normalized_text_embeds = nn.functional.normalize(text_embeds)
    return torch.mm(normalized_image_embeds, normalized_text_embeds.T)


class CustomStableDiffusionSafetyChecker(PreTrainedModel):
    config_class = CLIPConfig

    def __init__(self, config: CLIPConfig):
        super().__init__(config)

        self.vision_model = CLIPVisionModel(config.vision_config)
        self.visual_projection = nn.Linear(config.vision_config.hidden_size, config.projection_dim, bias=False)

        self.concept_embeds = nn.Parameter(torch.ones(17, config.projection_dim), requires_grad=False)
        self.special_care_embeds = nn.Parameter(torch.ones(3, config.projection_dim), requires_grad=False)

        self.register_buffer("concept_embeds_weights", torch.ones(17))
        self.register_buffer("special_care_embeds_weights", torch.ones(3))

    @torch.no_grad()
    def forward(self, clip_input, images):
        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        image_embeds = self.visual_projection(pooled_output)

        special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds).cpu().numpy()
        cos_dist = cosine_distance(image_embeds, self.concept_embeds).cpu().numpy()

        result = []
        batch_size = image_embeds.shape[0]
        for i in range(batch_size):
            result_img = {"special_scores": {}, "special_care": [], "concept_scores": {}, "bad_concepts": []}

            # increase this value to create a stronger `nfsw` filter
            # at the cost of increasing the possibility of filtering benign images
            adjustment = 0.0

            for concet_idx in range(len(special_cos_dist[0])):
                concept_cos = special_cos_dist[i][concet_idx]
                concept_threshold = self.special_care_embeds_weights[concet_idx].item()
                result_img["special_scores"][concet_idx] = round(concept_cos - concept_threshold + adjustment, 3)
                if result_img["special_scores"][concet_idx] > 0:
                    result_img["special_care"].append({concet_idx, result_img["special_scores"][concet_idx]})
                    adjustment = 0.01

            for concet_idx in range(len(cos_dist[0])):
                concept_cos = cos_dist[i][concet_idx]
                concept_threshold = self.concept_embeds_weights[concet_idx].item()
                result_img["concept_scores"][concet_idx] = round(concept_cos - concept_threshold + adjustment, 3)
                if result_img["concept_scores"][concet_idx] > 0:
                    result_img["bad_concepts"].append(concet_idx)

            result.append(result_img)

        has_nsfw_concepts = [len(res["bad_concepts"]) > 0 for res in result]

        # for idx, has_nsfw_concept in enumerate(has_nsfw_concepts):
        #     if has_nsfw_concept:
        #         images[idx] = np.zeros(images[idx].shape)  # black image

        # if any(has_nsfw_concepts):
        #     logger.warning(
        #         "Potential NSFW content was detected in one or more images. A black image will be returned instead."
        #         " Try again with a different prompt and/or seed."
        #     )

        return images, has_nsfw_concepts


class TextToImageModel:
    def __init__(self):
        self.model: Union[DiffusionPipeline, None] = None
        os.makedirs(model_settings.model_output_path, exist_ok=True)
        self.load_model()

    def load_model(self) -> None:
        if torch.cuda.is_available():
            if model_settings.model_class_name == ModelClassNameEnums.STABLE_DIFFUSION:
                self.diffusion_pipeline = StableDiffusionPipeline.from_pretrained(
                    model_settings.model_name_or_path, torch_dtype=torch.float16
                ).to("cuda")
                self.diffusion_pipeline.safety_checker = CustomStableDiffusionSafetyChecker.from_pretrained(
                    os.path.join(model_settings.model_name_or_path, "safety_checker"), torch_dtype=torch.float16
                ).to("cuda")
            else:
                self.diffusion_pipeline = DiffusionPipeline.from_pretrained(
                    model_settings.model_name_or_path, torch_dtype=torch.float16
                ).to("cuda")
        else:
            logger.error("CPU Mode is not Supported")
            exit(1)

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

        gpu_info = nvgpu.gpu_info()[0]
        mem_free = gpu_info["mem_total"] - gpu_info["mem_used"]
        estimated_memory_usage_per_image = (
            model_settings.model_unit_memory * ((data.width / 512) ** 2) * ((data.height / 512) ** 2)
        )
        if estimated_memory_usage_per_image * data.images >= mem_free:
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
                    images.append(image)
                    filter_results.append(filter_result)
                    base_seed_list.append(seed)
                    image_no_list.append(1)
                clear_memory()
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
            clear_memory()
        output_path = os.path.join(model_settings.model_output_path, task_id)
        os.makedirs(output_path, exist_ok=True)

        result: List[ImageGenerationWorkerOutput] = []
        for i in range(data.images):
            if filter_results[i]:
                images[i].save(os.path.join(output_path, f"{i + 1}_origin.png"))
                images[i] = Image.new(mode="RGB", size=images[i].size)
            images[i].save(os.path.join(output_path, f"{i + 1}.png"))
            result.append(
                ImageGenerationWorkerOutput(
                    image_path=os.path.join(output_path, f"{i + 1}.png"),
                    origin_image_path=os.path.join(output_path, f"{i + 1}_origin.png") if filter_results[i] else None,
                    nsfw_content_detected=filter_results[i],
                    base_seed=base_seed_list[i],
                    image_no=image_no_list[i],
                )
            )
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

        return result

import os
from typing import Union

import numpy as np
import open_clip
import torch
from configs.config import model_settings
from constants import CONFIG_FILE_NAME, MODEL_FILE_NAME, OUT_DIR
from einops import rearrange
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image
from torchvision.utils import make_grid


class TextToImageModel:
    def __init__(self):
        self.clip_model = None
        self.model: Union[LatentDiffusion, None] = None
        self.preprocess = None
        os.makedirs(OUT_DIR, exist_ok=True)
        sample_path = os.path.join(OUT_DIR, "samples")
        os.makedirs(sample_path, exist_ok=True)

    def load_model(self) -> None:
        config = OmegaConf.load(os.path.join(model_settings.model_path, CONFIG_FILE_NAME))
        pl_sd = torch.load(os.path.join(model_settings.model_path, MODEL_FILE_NAME))
        sd = pl_sd["state_dict"]
        self.model = instantiate_from_config(config.model)
        m, u = self.model.load_state_dict(sd, strict=False)
        if len(m) > 0:
            logger.warning(f"Missing Keys : {m}")
        if len(u) > 0:
            logger.warning(f"Unexpected Keys : {u}")
        if torch.cuda.is_available():
            self.model = self.model.half().cuda()
        self.model.eval()

    def load_clip_model(self) -> None:
        clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        logger.info(type(clip_model))
        self.clip_model = clip_model
        self.preprocess = preprocess

    def generate(self, opt, task_id) -> str:
        if opt.plms:
            opt.ddim_eta = 0
            sampler = PLMSSampler(self.model)
        else:
            sampler = DDIMSampler(self.model)

        outpath = OUT_DIR
        prompt = opt.prompt

        all_samples = list()
        all_samples_images = list()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                with self.model.ema_scope():
                    uc = None
                    if opt.scale > 0:
                        uc = self.model.get_learned_conditioning(opt.n_samples * [""])
                    for n in range(opt.n_iter):
                        c = self.model.get_learned_conditioning(opt.n_samples * [prompt])
                    shape = [4, opt.H // 8, opt.W // 8]
                    samples_ddim, _ = sampler.sample(
                        S=opt.ddim_steps,
                        conditioning=c,
                        batch_size=opt.n_samples,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=opt.scale,
                        unconditional_conditioning=uc,
                        eta=opt.ddim_eta,
                    )

                    x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                    for x_sample in x_samples_ddim:
                        x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), "c h w -> h w c")
                        image_vector = Image.fromarray(x_sample.astype(np.uint8))
                        image_preprocess = self.preprocess(image_vector).unsqueeze(0)
                        with torch.no_grad():
                            image_features = self.clip_model.encode_image(image_preprocess)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        all_samples_images.append(image_vector)
                    all_samples.append(x_samples_ddim)

                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, "n b c h w -> (n b) c h w")
                    grid = make_grid(grid, nrow=2)
                    # to image
                    grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()

                    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f"{task_id}.png"))
                    return os.path.join(outpath, f"{task_id}.png")

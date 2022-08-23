import argparse
import json
import os
from datetime import datetime
from typing import Dict

import numpy as np
import pytz
import torch
from celery.signals import celeryd_init
from einops import rearrange
from enums import ResponseStatusEnum
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.models.diffusion.plms import PLMSSampler
from loguru import logger
from ml_model import TextToImageModel
from payloads.response import ImageGenerationResponse
from PIL import Image
from torchvision.utils import make_grid
from utils import clear_memory
from worker import app, redis


tti = TextToImageModel()


@celeryd_init.connect
def load_model(**kwargs):
    logger.info("Start loading model...")
    tti.load_model()
    tti.load_clip_model()

    logger.info("Loading model is done!")


@app.task(name="generate")
def generate(task_id: str, data: Dict) -> None:
    now = datetime.utcnow().replace(tzinfo=pytz.utc).timestamp()
    response = ImageGenerationResponse(status=ResponseStatusEnum.ASSIGNED, updated_at=now)
    redis.set(task_id, json.dumps(dict(response)))

    model: LatentDiffusion = tti.model
    clip_model = tti.clip_model
    preprocess = tti.preprocess
    logger.info(type(clip_model))
    opt = argparse.Namespace(
        prompt=data.prompt,
        outdir="latent-diffusion/outputs",
        ddim_steps=data.steps,
        ddim_eta=0,
        n_iter=1,
        W=data.width,
        H=data.height,
        n_samples=int(data.images),
        scale=data.diversity_scale,
        plms=True,
    )

    if opt.plms:
        opt.ddim_eta = 0
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    prompt = opt.prompt

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)

    all_samples = list()
    all_samples_images = list()

    error_flag = False
    try:
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                with model.ema_scope():
                    uc = None
                    if opt.scale > 0:
                        uc = model.get_learned_conditioning(opt.n_samples * [""])
                    for n in range(opt.n_iter):
                        c = model.get_learned_conditioning(opt.n_samples * [prompt])
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

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                    for x_sample in x_samples_ddim:
                        x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), "c h w -> h w c")
                        image_vector = Image.fromarray(x_sample.astype(np.uint8))
                        image_preprocess = preprocess(image_vector).unsqueeze(0)
                        with torch.no_grad():
                            image_features = clip_model.encode_image(image_preprocess)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        all_samples_images.append(image_vector)
                    all_samples.append(x_samples_ddim)
    except ValueError as e:
        redis.set(task_id, json.dumps({"status_code": 422, "message": str(e)}))
        error_flag = True
    except Exception as e:
        redis.set(task_id, json.dumps({"status_code": 500, "message": str(e)}))
        error_flag = True
    finally:
        clear_memory()
    if error_flag:
        pass
    else:
        # additionally, save as grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, "n b c h w -> (n b) c h w")
        grid = make_grid(grid, nrow=2)
        # to image
        grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()

        now = datetime.utcnow().replace(tzinfo=pytz.utc).timestamp()
        response.updated_at = now
        response.status = ResponseStatusEnum.COMPLETED
        logger.info(f"task_id: {task_id}, gen result: {response.result}")
        redis.set(task_id, json.dumps(dict(response)))
        Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f"{task_id}.png"))
        return os.path.join(outpath, f"{task_id}.png")

import os

import open_clip
import torch
from configs.config import model_settings
from constants import CONFIG_FILE_NAME, MODEL_FILE_NAME
from ldm.util import instantiate_from_config
from loguru import logger
from omegaconf import OmegaConf


class TextToImageModel:
    def __init__(self):
        self.clip_model = None
        self.model = None
        self.preprocess = None

    def load_model(self) -> None:
        logger.info("Load Model")
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
        self.clip_model = clip_model
        self.preprocess = preprocess

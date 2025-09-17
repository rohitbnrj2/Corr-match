"""
Perform Diffusion based Inpainting on SPAD Data. This module is built to finetune
a pre-trained diffusion model to generate Luminance images from SPAD data.
"""
from __future__ import annotations
from typing import Sequence, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn

import diffusers
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers", 
    token="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", 
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

image = pipe(
    "A cat holding a sign that says hello world",
    negative_prompt="",
    num_inference_steps=28,
    guidance_scale=7.0,
).images[0]
image



"""Photoreal style bridge: nudge the masked figure toward natural-photo
distribution before SAM 3D, which was trained on real photographs and
degrades on heavily stylized inputs (Monet, Van Gogh, etc.).

We run SDXL img2img at low denoise strength so identity / composition is
preserved but brushstrokes are smoothed into something the lift model
recognizes. Only the masked region is replaced — background is untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


def photoreal_bridge(
    image_path: Path,
    mask: np.ndarray,
    prompt: str,
    strength: float = 0.4,
    seed: int = 42,
) -> Image.Image:
    """Return an RGB PIL image with the masked region nudged photoreal."""
    import torch
    from diffusers import StableDiffusionXLImg2ImgPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        variant="fp16" if device == "cuda" else None,
    ).to(device)

    base = Image.open(image_path).convert("RGB")
    init = base.copy()

    generator = torch.Generator(device=device).manual_seed(seed)
    photoreal = pipe(
        prompt=f"photorealistic portrait of {prompt}, natural lighting, sharp focus",
        negative_prompt="painting, illustration, brushstrokes, cartoon, low quality",
        image=init,
        strength=strength,
        guidance_scale=6.5,
        num_inference_steps=28,
        generator=generator,
    ).images[0]

    # Composite: photoreal where mask is True, original elsewhere.
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    composited = Image.composite(photoreal, base, mask_img)
    return composited


def save_image(img: Image.Image, path: Path) -> None:
    img.save(path)

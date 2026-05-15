"""Segment a figure out of a painting using SAM 3 (text-promptable).

If `mask` is supplied on the request we skip segmentation and use it directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


def load_mask(path: Path) -> np.ndarray:
    """Load a binary mask PNG → bool array (H, W)."""
    arr = np.array(Image.open(path).convert("L"))
    return arr > 127


def segment_with_sam3(image_path: Path, prompt: str) -> np.ndarray:
    """Run SAM 3 with a text prompt and return the best mask.

    SAM 3's text-promptable segmentation is the cleanest UX here — one
    prompt like "person" or "boat" picks the figure without a separate
    detector. Requires the `sam3` package + checkpoint baked into the
    runpod image (see runpod/Dockerfile).
    """
    # Imported lazily so CLI clients without a GPU stack still work.
    import torch
    from sam3 import Sam3Predictor  # type: ignore[import-not-found]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = Sam3Predictor.from_pretrained("facebook/sam3", device=device)

    image = Image.open(image_path).convert("RGB")
    result = predictor.predict_from_text(image=image, text=prompt)

    # SAM 3 returns ranked masks; pick the highest-confidence one.
    masks = result["masks"]
    scores = result["scores"]
    best = int(np.argmax(scores))
    return masks[best].astype(bool)


def get_mask(
    image_path: Path,
    prompt: str,
    explicit_mask: Optional[Path] = None,
) -> np.ndarray:
    if explicit_mask is not None:
        return load_mask(explicit_mask)
    return segment_with_sam3(image_path, prompt)


def save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)

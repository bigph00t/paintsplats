"""Lift a masked 2D figure into a 3D Gaussian Splat using SAM 3D Objects.

Reference: https://github.com/facebookresearch/sam-3d-objects
Output: a .ply Gaussian splat (millions of view-dependent Gaussians).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


def lift_to_splat(
    image: Image.Image | Path,
    mask: np.ndarray,
    out_path: Path,
    seed: int = 42,
    pipeline_yaml: str = "checkpoints/hf/pipeline.yaml",
) -> Path:
    """Run SAM 3D Objects and write a .ply splat to `out_path`."""
    # Imports gated so CLI clients on machines without CUDA can still
    # call the serverless endpoint without installing torch.
    from sam_3d_objects.inference import (  # type: ignore[import-not-found]
        Inference,
        load_image,
        load_single_mask,
    )

    inference = Inference(pipeline_yaml, compile=False)

    if isinstance(image, Image.Image):
        # SAM 3D Objects' loader expects a path; round-trip via temp file
        # only if necessary. Most callers pass a path already.
        tmp = out_path.with_suffix(".prep.png")
        image.save(tmp)
        image_tensor = load_image(str(tmp))
    else:
        image_tensor = load_image(str(image))

    mask_tensor = load_single_mask(mask)

    output = inference(image_tensor, mask_tensor, seed=seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output["gs"].save_ply(str(out_path))
    return out_path


def count_splats(ply_path: Path) -> Optional[int]:
    """Cheap header parse: report the vertex count from a binary .ply."""
    try:
        with open(ply_path, "rb") as f:
            header = []
            while True:
                line = f.readline()
                header.append(line)
                if line.strip() == b"end_header" or len(header) > 200:
                    break
        for line in header:
            if line.startswith(b"element vertex"):
                return int(line.split()[-1])
    except Exception:
        return None
    return None

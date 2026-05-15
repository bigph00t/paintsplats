"""End-to-end pipeline orchestration.

This module is the single source of truth for the ordering and is shared
by both the local CLI (`paintsplats local`) and the RunPod serverless
handler (`runpod/handler.py`).
"""

from __future__ import annotations

import time
from pathlib import Path

from .lift import count_splats, lift_to_splat
from .prep import photoreal_bridge, save_image
from .segment import get_mask, save_mask
from .types import PipelineRequest, PipelineResult


def run(req: PipelineRequest, work_dir: Path | None = None) -> PipelineResult:
    work_dir = work_dir or req.out.parent
    work_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    mask = get_mask(req.image, req.prompt, explicit_mask=req.mask)
    mask_path = work_dir / f"{req.out.stem}.mask.png"
    save_mask(mask, mask_path)

    prep_path = None
    image_for_lift: Path = req.image
    if req.photoreal_prep:
        photoreal = photoreal_bridge(
            req.image, mask, prompt=req.prompt,
            strength=req.prep_strength, seed=req.seed,
        )
        prep_path = work_dir / f"{req.out.stem}.prep.png"
        save_image(photoreal, prep_path)
        image_for_lift = prep_path

    ply_path = lift_to_splat(
        image=image_for_lift,
        mask=mask,
        out_path=req.out,
        seed=req.seed,
    )

    return PipelineResult(
        ply_path=ply_path,
        mask_path=mask_path,
        prep_path=prep_path,
        elapsed_seconds=time.perf_counter() - started,
        splat_count=count_splats(ply_path),
    )

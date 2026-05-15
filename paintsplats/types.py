"""Shared data types for the Paintsplats pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


SegmentBackend = Literal["sam3", "manual"]


@dataclass
class PipelineRequest:
    image: Path
    prompt: str = "person"
    mask: Optional[Path] = None
    seed: int = 42
    photoreal_prep: bool = True
    prep_strength: float = 0.4
    out: Path = field(default_factory=lambda: Path("splat.ply"))
    segment_backend: SegmentBackend = "sam3"


@dataclass
class PipelineResult:
    ply_path: Path
    mask_path: Optional[Path] = None
    prep_path: Optional[Path] = None
    elapsed_seconds: float = 0.0
    splat_count: Optional[int] = None

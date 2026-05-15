"""RunPod serverless handler.

Input JSON:
  {
    "image_url": "https://...",       # OR image_b64
    "image_b64": "...",
    "prompt": "person",
    "seed": 42,
    "photoreal_prep": true,
    "prep_strength": 0.4
  }

Output JSON (one of):
  { "ply_url": "https://...presigned" }    # when S3 env is configured
  { "ply_b64": "..." }                     # otherwise, inline
"""

from __future__ import annotations

import base64
import os
import tempfile
import uuid
from pathlib import Path

import requests
import runpod  # type: ignore[import-not-found]

from paintsplats.pipeline import run as run_pipeline
from paintsplats.types import PipelineRequest

# Surface HF token to huggingface_hub regardless of which name the user set.
_hf = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
if _hf:
    os.environ.setdefault("HF_TOKEN", _hf)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", _hf)


def _materialize_image(job_input: dict, work_dir: Path) -> Path:
    target = work_dir / "input.png"
    if url := job_input.get("image_url"):
        target.write_bytes(requests.get(url, timeout=60).content)
    elif b64 := job_input.get("image_b64"):
        target.write_bytes(base64.b64decode(b64))
    else:
        raise ValueError("input requires image_url or image_b64")
    return target


def _maybe_upload(ply_path: Path) -> dict:
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        return {"ply_b64": base64.b64encode(ply_path.read_bytes()).decode()}

    import boto3  # type: ignore[import-not-found]
    from botocore.config import Config

    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT") or None,
        region_name=os.getenv("S3_REGION", "auto"),
        aws_access_key_id=os.getenv("S3_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4"),
    )
    key = f"paintsplats/{uuid.uuid4()}.ply"
    s3.upload_file(str(ply_path), bucket, key, ExtraArgs={"ContentType": "model/ply"})
    url = s3.generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=86_400
    )
    return {"ply_url": url}


def handler(job: dict) -> dict:
    job_input = job.get("input") or {}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        image = _materialize_image(job_input, work)
        out = work / "splat.ply"

        req = PipelineRequest(
            image=image,
            prompt=job_input.get("prompt", "person"),
            seed=int(job_input.get("seed", 42)),
            photoreal_prep=bool(job_input.get("photoreal_prep", True)),
            prep_strength=float(job_input.get("prep_strength", 0.4)),
            out=out,
        )
        result = run_pipeline(req, work_dir=work)

        response = _maybe_upload(result.ply_path)
        response["splat_count"] = result.splat_count
        response["elapsed_seconds"] = result.elapsed_seconds
        return response


runpod.serverless.start({"handler": handler})

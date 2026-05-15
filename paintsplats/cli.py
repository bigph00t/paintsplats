"""paintsplats CLI.

  paintsplats remote monet.jpg --prompt "boat" --out boat.ply
  paintsplats local  monet.jpg --prompt "boat" --out boat.ply   # needs GPU
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


@app.callback()
def _bootstrap() -> None:
    load_dotenv()


@app.command()
def local(
    image: Path = typer.Argument(..., exists=True, readable=True),
    prompt: str = typer.Option("person", help="Text describing the figure to extract."),
    mask: Optional[Path] = typer.Option(None, exists=True, help="Skip segmentation; use this mask."),
    out: Path = typer.Option(Path("splat.ply"), help="Output .ply path."),
    seed: int = typer.Option(42),
    no_prep: bool = typer.Option(False, "--no-prep", help="Skip the photoreal style bridge."),
    prep_strength: float = typer.Option(0.4, min=0.0, max=1.0),
) -> None:
    """Run the full pipeline locally. Requires a CUDA GPU and the `gpu` extras."""
    from .pipeline import run
    from .types import PipelineRequest

    req = PipelineRequest(
        image=image, prompt=prompt, mask=mask, seed=seed,
        photoreal_prep=not no_prep, prep_strength=prep_strength, out=out,
    )
    console.print(f"[bold]paintsplats[/] local · prompt={prompt!r} prep={not no_prep}")
    result = run(req)
    console.print(
        f"[green]done[/] {result.ply_path} "
        f"({result.splat_count or '?'} splats, {result.elapsed_seconds:.1f}s)"
    )


@app.command()
def remote(
    image: Path = typer.Argument(..., exists=True, readable=True),
    prompt: str = typer.Option("person"),
    out: Path = typer.Option(Path("splat.ply")),
    seed: int = typer.Option(42),
    no_prep: bool = typer.Option(False, "--no-prep"),
    endpoint: Optional[str] = typer.Option(None, envvar="RUNPOD_ENDPOINT_ID"),
    api_key: Optional[str] = typer.Option(None, envvar="RUNPOD_API_KEY"),
) -> None:
    """Invoke the RunPod serverless endpoint and write the splat locally."""
    import requests

    if not endpoint or not api_key:
        raise typer.BadParameter("Set RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY in your env or .env")

    payload = {
        "input": {
            "image_b64": base64.b64encode(image.read_bytes()).decode(),
            "prompt": prompt,
            "seed": seed,
            "photoreal_prep": not no_prep,
        }
    }
    url = f"https://api.runpod.ai/v2/{endpoint}/runsync"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    console.print(f"[bold]paintsplats[/] remote → {endpoint}")
    t0 = time.perf_counter()
    resp = requests.post(url, json=payload, headers=headers, timeout=600)
    resp.raise_for_status()
    body = resp.json()
    job_output = body.get("output") or {}

    if "ply_url" in job_output:
        ply_bytes = requests.get(job_output["ply_url"], timeout=120).content
    elif "ply_b64" in job_output:
        ply_bytes = base64.b64decode(job_output["ply_b64"])
    else:
        raise RuntimeError(f"Endpoint returned no splat: {body}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(ply_bytes)
    console.print(f"[green]done[/] {out} ({len(ply_bytes)/1_048_576:.1f} MB, {time.perf_counter()-t0:.1f}s)")


@app.command()
def version() -> None:
    from . import __version__
    console.print(__version__)


if __name__ == "__main__":
    app()

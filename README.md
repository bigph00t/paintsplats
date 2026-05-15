# Paintsplats

> Pull a 3D Gaussian Splat of a figure out of a painting. Point at a Monet, get a `.ply` of the boatman.

Paintsplats is a small wrapper around **Meta SAM 3** (segmentation) and **Meta SAM 3D Objects** (single-image → Gaussian splat) tuned for the awkward case where the input is a painting rather than a photo. A short SDXL img2img pass bridges the style gap before the lift so the splat actually looks like the figure.

The pipeline runs as a **RunPod Serverless** endpoint that you deploy on your own account. Cost at hobby scale (~50 runs/month on an L40S) is a few dollars; idle cost is zero.

---

## Pipeline

```
painting.jpg ──► SAM 3 (text prompt)  ──► mask
                                          │
                                          ▼
                        SDXL img2img (strength 0.4, masked region only)
                                          │
                                          ▼
                                 SAM 3D Objects
                                          │
                                          ▼
                                     splat.ply
```

Each stage is in its own module — swap any of them.

| Stage      | Module                    | What it does                                                                          |
|------------|---------------------------|---------------------------------------------------------------------------------------|
| Segment    | `paintsplats/segment.py`  | SAM 3 with a text prompt (`"person"`, `"boat"`). Or supply your own mask PNG.         |
| Style bridge | `paintsplats/prep.py`   | SDXL img2img on the masked crop. SAM 3D was trained on photos; this de-paints it.    |
| Lift       | `paintsplats/lift.py`     | SAM 3D Objects. Writes a `.ply` Gaussian splat directly via `output["gs"].save_ply`.  |

---

## Quickstart — hit a remote endpoint

After you deploy your own endpoint (see below), the CLI is the only piece you need locally:

```bash
pip install paintsplats
cp .env.example .env  # fill in RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID
paintsplats remote monet-boating.jpg --prompt "boatman" --out boatman.ply
```

Open `boatman.ply` in [SuperSplat](https://superspl.at/editor) (free, browser-based) to orbit the figure.

---

## Self-deploy on RunPod

You'll need:

- A Hugging Face account with access to [`facebook/sam-3d-objects`](https://huggingface.co/facebook/sam-3d-objects) (gated — apply on the model page).
- A RunPod account and API key.
- ~25 GB of free space for the Docker image.

```bash
# 1. Build with HF_TOKEN baked in as a BuildKit secret (NOT in the image).
export HF_TOKEN=hf_xxx
DOCKER_BUILDKIT=1 docker build \
  --secret id=hf,env=HF_TOKEN \
  -t your-dockerhub/paintsplats:0.1.0 \
  -f runpod/Dockerfile .

# 2. Push.
docker push your-dockerhub/paintsplats:0.1.0

# 3. In the RunPod console: Serverless → New Endpoint
#    - Container image: your-dockerhub/paintsplats:0.1.0
#    - GPU: L40S 48GB (recommended) or A40 / A6000 48GB
#    - Container disk: 40 GB
#    - Idle timeout: 5s   (hobby-friendly; FlashBoot covers warm restarts)
#    - Max workers: 1     (raise if you have traffic)
#    - Env vars: any S3_* if you want presigned URLs instead of inline base64.
```

Save the new endpoint ID into `.env` as `RUNPOD_ENDPOINT_ID` and you're done.

### GPU sizing

| GPU              | VRAM   | Active rate*       | Notes                                                       |
|------------------|--------|--------------------|-------------------------------------------------------------|
| RTX 4090         | 24 GB  | ~$1.12/hr          | Tight. SAM 3D officially wants 32 GB.                       |
| A40              | 48 GB  | ~$1.22/hr          | Cheapest comfortable fit.                                   |
| A6000            | 48 GB  | ~$1.22/hr          | Same tier, sometimes better availability.                   |
| **L40S**         | 48 GB  | ~$1.91/hr          | ~2× throughput of A40. **Recommended.**                      |

*Approximate RunPod serverless flex rates as of May 2026; check current pricing.*

---

## Local run (skip RunPod)

If you have a 32 GB+ GPU, you can run end-to-end locally:

```bash
pip install -e ".[gpu]"
paintsplats local monet-boating.jpg --prompt "boatman" --out boatman.ply
```

Add `--no-prep` to skip the SDXL style bridge (faster, but worse results on stylized inputs).

---

## Roadmap

- [ ] **v0.1** — pipeline scaffolding (this commit)
- [ ] **v0.2** — verified end-to-end run on Monet, Hopper, Van Gogh; example splats checked into `examples/`
- [ ] **v0.3** — route human figures through HumanSplat for better anatomy
- [ ] **v0.4** — web demo (drag-drop painting → splat URL)
- [ ] **v0.5** — batch mode + caching by `(image_hash, prompt)`

---

## License

Wrapper code: MIT. **Model weights** carry their own (mostly non-commercial) licenses — see `LICENSE` for the full caveat. Paintsplats is designed for self-deployment; if you want to ship a hosted commercial product, read the upstream model licenses first.

## Acknowledgements

- Meta AI — [SAM 3](https://github.com/facebookresearch/sam3), [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects)
- Stability AI — Stable Diffusion XL
- [RunPod](https://www.runpod.io/) Serverless

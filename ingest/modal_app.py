"""Modal serverless deployment for the acq-search-v2 FastAPI backend.

Design goal: live behavior == local behavior, byte-for-byte.

  - Same FastAPI app (api.main)
  - Same hybrid scoring code (lib.hybrid, lib.reranker, lib.query_parser, lib.structural)
  - Same data files (ingest/cache/* bundled into the image, unmodified)
  - Same CLIP model (ViT-L-14 / LAION2B-32B-B82K, baked into the image at build time)
  - Same OpenAI key reaches parser + reranker via Modal Secret

Container layout mirrors the local repo so config.py's PROJECT_ROOT math
resolves the same way it does on a laptop:

  local                              modal container
  ----------------------------       --------------------------------
  acq_search_v2/                     /root/app/
    ingest/                            ingest/
      config.py                          config.py
      cache/                             cache/
      lib/                               lib/
      api/main.py                        api/main.py

Deploy
  pip install modal                     # one-time
  modal token new                       # one-time interactive auth
  modal secret create openai-key \
      OPENAI_API_KEY=sk-...             # one-time
  modal deploy ingest/modal_app.py      # publishes; outputs a stable URL

Cost
  Modal: $30/mo free credit; CPU-only and scale-to-zero — realistic burn ~$0
  during demo. OpenAI: ~$0.001 per query (parser + reranker).
"""
import modal

# /root/app is the in-container repo root. Everything below mirrors the local
# layout so config.py's `parent.parent` arithmetic still points at the repo root.
REPO_ROOT_REMOTE = "/root/app"
INGEST_REMOTE = f"{REPO_ROOT_REMOTE}/ingest"

# CLIP model identifiers — must match lib/hybrid.py MODEL_NAME / PRETRAINED.
CLIP_MODEL_NAME = "ViT-L-14"
CLIP_PRETRAINED = "laion2b_s32b_b82k"


def _bake_clip_model():
    """Download the CLIP weights at image-build time so cold starts don't pay
    a HuggingFace round-trip (and don't depend on HF Hub availability)."""
    import open_clip
    open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device="cpu"
    )
    # also pull the tokenizer so its small file lands in the cache
    open_clip.get_tokenizer(CLIP_MODEL_NAME)


image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        # Web framework
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.30.0",
        "pydantic>=2.7.0",
        # Numerics
        "numpy>=1.26.0",
        # Vision (CLIP)
        "open_clip_torch>=2.30.0",
        "Pillow>=10.4.0",
        # LLM client used by parser + reranker
        "openai>=1.40.0",
        # config.py uses dotenv; harmless if .env isn't present
        "python-dotenv>=1.0.0",
    )
    # Pre-download the CLIP weights into the image. This runs ONCE during image
    # build and is cached across deploys; cold starts get a fully-warm cache.
    .run_function(_bake_clip_model)
    # Mirror the local repo layout under /root/app/ingest/* so config.py and
    # the lib/api imports see the same paths they see on the laptop.
    .add_local_dir("ingest/cache", remote_path=f"{INGEST_REMOTE}/cache", copy=True)
    .add_local_dir("ingest/lib",   remote_path=f"{INGEST_REMOTE}/lib",   copy=True)
    .add_local_dir("ingest/api",   remote_path=f"{INGEST_REMOTE}/api",   copy=True)
    .add_local_file("ingest/config.py", remote_path=f"{INGEST_REMOTE}/config.py", copy=True)
)

app = modal.App("acq-search-v2-backend")


@app.function(
    image=image,
    cpu=2,
    memory=4096,                # CLIP + indexes need ~3 GB
    timeout=300,
    min_containers=0,           # scale-to-zero when idle
    max_containers=4,
    scaledown_window=300,       # 5 min warm window after last request
    secrets=[modal.Secret.from_name("openai-key")],  # mounts OPENAI_API_KEY into env
)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def fastapi_app():
    """ASGI entrypoint Modal serves. Imports the existing FastAPI app unchanged."""
    import sys
    # Match local: sys.path includes the ingest/ dir so `from lib.hybrid import ...`,
    # `from api.main import app`, etc. resolve exactly like they do on a laptop.
    if INGEST_REMOTE not in sys.path:
        sys.path.insert(0, INGEST_REMOTE)
    from api.main import app as fastapi_application
    return fastapi_application

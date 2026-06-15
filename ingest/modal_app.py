"""Modal serverless deployment for the acq-search-v2 FastAPI backend.

Why Modal:
- Scales to zero (no idle cost)
- Free $30/mo credit covers the demo period
- Cold start ~15-30s, warm requests <100ms
- Persistent stable URL after first deploy
- Same Python codebase as local dev — no rewrite

Deploy
  pip install modal
  modal token new                 # one-time interactive auth
  modal deploy ingest/modal_app.py
  # outputs the stable URL — set as BACKEND_URL in Vercel

Cost
  $30 free credit per month. This service is idle most of the time and uses
  CPU only. Realistic cost during demo period: ~$0 — well under the free tier.
"""
import modal

ROOT = "/root/app"  # where we'll mount everything inside the container

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.30.0",
        "pydantic>=2.7.0",
        "numpy>=1.26.0",
        "open_clip_torch>=2.30.0",
        "Pillow>=10.4.0",
    )
    # bundle the cache + code into the image. Modal's container builds once and reuses
    # across cold starts; only changed local files trigger rebuilds.
    .add_local_dir("ingest/cache", remote_path=f"{ROOT}/cache", copy=True)
    .add_local_dir("ingest/lib", remote_path=f"{ROOT}/lib", copy=True)
    .add_local_dir("ingest/api", remote_path=f"{ROOT}/api", copy=True)
    .add_local_file("ingest/config.py", remote_path=f"{ROOT}/config.py", copy=True)
)

app = modal.App("acq-search-v2-backend")


@app.function(
    image=image,
    cpu=2,
    memory=4096,            # CLIP + index needs ~3 GB
    timeout=300,
    min_containers=0,       # scale to zero when idle
    max_containers=4,
    scaledown_window=300,   # keep warm for 5 min after last request
)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def fastapi_app():
    """ASGI entrypoint Modal serves. Imports our existing FastAPI app."""
    import sys
    sys.path.insert(0, ROOT)
    from api.main import app as fastapi_application
    return fastapi_application

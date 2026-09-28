"""
backend.main
~~~~~~~~~~~~
FastAPI application entry point.
Trains ML models once on startup, then serves the API on port 8000.

Run:
    python -m backend.main
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import asyncio
from app.ml.trainer import train_all
from backend.api import predict as predict_module
from backend.api.predict import router as predict_router
from backend.api.diseases import router as diseases_router
from backend.api.user import router as user_router
from backend.services.knowledge_updater import background_sync_worker


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Train all XGBoost models once at startup and start 24-hour knowledge sync worker."""
    print("[*] Training ML models on startup...")
    models = train_all(verbose=False)
    predict_module.set_models(models)
    print(f"[OK] Models ready: {list(models.keys())}")

    print("[*] Launching 24-hour Disease Knowledge Base sync worker...")
    sync_task = asyncio.create_task(background_sync_worker(interval_seconds=3600))

    yield

    print("[*] Cancelling background sync worker...")
    sync_task.cancel()
    try:
        await sync_task
    except asyncio.CancelledError:
        pass
    print("[*] Shutting down cleanly.")


app = FastAPI(
    title="MediPredict AI — Backend API",
    description="Multi-disease risk prediction powered by XGBoost + SHAP.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS: allow Vite dev, preview, and GitHub Pages ─────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "https://vinila-velmala.github.io",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────
PREFIX = "/api/v1"
app.include_router(predict_router,  prefix=PREFIX, tags=["Prediction"])
app.include_router(diseases_router, prefix=PREFIX, tags=["Diseases"])
app.include_router(user_router,     prefix=PREFIX, tags=["User"])


@app.get("/")
async def health():
    return {"status": "ok", "service": "MediPredict AI Backend"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)

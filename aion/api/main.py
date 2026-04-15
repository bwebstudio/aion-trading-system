"""
aion.api.main
──────────────
FastAPI application factory.

STUB — not yet implemented.

Start with: uvicorn aion.api.main:app --reload
"""

from fastapi import FastAPI

app = FastAPI(
    title="AION Trading AI Platform",
    version="0.1.0",
    description="API for monitoring and controlling the AION trading system.",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": "0.1.0"}

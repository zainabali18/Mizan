"""FastAPI service.

Run locally:   uvicorn mizan.api:app --port 8000      then open http://localhost:8000/docs
Run in Docker: see Dockerfile

Privacy: the raw request text is never logged or returned; responses contain
only the PII-redacted text.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .service import MizanClassifier


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000,
                      examples=["بطاقتي 4111 1111 1111 1111 انسرقت، شو اعمل؟"])


class IntentScore(BaseModel):
    intent: str
    confidence: float


class ClassifyResponse(BaseModel):
    intent: str
    confidence: float
    route: str = Field(..., description='"auto" = answer automatically, "human" = send to an agent')
    threshold: float
    top_intents: list[IntentScore]
    redacted_text: str
    pii_detected: list[str]
    latency_ms: float


def create_app(classifier: MizanClassifier | None = None) -> FastAPI:
    clf = classifier or MizanClassifier()
    app = FastAPI(
        title="Mizan API",
        version=__version__,
        description="Sovereign Arabic intent classification for banking support, "
                    "with PII redaction and human handoff.",
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": clf.config["model_id"], "confidence_threshold": clf.threshold}

    @app.get("/intents")
    def intents() -> list[str]:
        return clf.intents

    @app.post("/classify", response_model=ClassifyResponse)
    def classify(req: ClassifyRequest) -> dict:
        try:
            return clf.classify(req.text)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    return app


app = create_app()

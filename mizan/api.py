"""FastAPI service.

Run locally:   uvicorn mizan.api:app --port 8000      then open http://localhost:8000/docs
Run in Docker: see Dockerfile

Privacy: the raw request text is never logged or returned. Successful responses
contain only the PII-redacted text, and validation errors (empty, too long,
malformed) say what was wrong without echoing the input back.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
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
    route: str = Field(..., description='"auto" = score meets the threshold; "human" = flagged for human review')
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
        description="Self-hostable Arabic intent classification for banking support. Personal data is "
                    "redacted before classification, and requests scoring below the threshold are flagged "
                    "for human review (the flag is returned; no review queue is connected).",
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default 422 body includes the submitted value ("input"), which could be a
        # customer message containing personal data. Return what was wrong, never the value itself.
        errors = [{k: v for k, v in e.items() if k in ("type", "loc", "msg")} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": errors})

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
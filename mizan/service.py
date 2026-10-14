"""The production classifier: raw customer text in, a safe routing decision out.

    clf = MizanClassifier()
    clf.classify("بطاقتي 4111 1111 1111 1111 انسرقت")
    -> {"intent": "...", "confidence": 0.97, "route": "auto",
        "redacted_text": "بطاقتي [CARD_NUMBER] انسرقت", "pii_detected": ["CARD_NUMBER"], ...}

Guarantees:
  * PII is redacted BEFORE the text reaches the model, and only redacted text is returned.
  * Requests below the confidence threshold are routed to a human ("route": "human").
  * The model is loaded lazily from the Hugging Face Hub (or a local folder) and runs on CPU.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from .pii import redact

ROOT = Path(__file__).resolve().parent.parent
# MIZAN_CONFIG lets tests/deployments point at another config without code changes.
CONFIG_PATH = Path(os.environ.get("MIZAN_CONFIG", ROOT / "mizan_config.json"))


class Backend(Protocol):
    id2label: dict[int, str]

    def __call__(self, texts: list[str]) -> np.ndarray: ...


class TransformersBackend:
    """Wraps a fine-tuned Hugging Face sequence classifier (CPU by default)."""

    def __init__(self, model_id: str, max_length: int = 64, device: str = "cpu"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device).eval()
        self.device = device
        self.max_length = max_length
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    def __call__(self, texts: list[str]) -> np.ndarray:
        enc = self.tokenizer(texts, truncation=True, max_length=self.max_length,
                             padding=True, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            logits = self.model(**enc).logits
        return self._torch.softmax(logits, dim=-1).cpu().numpy()


class MizanClassifier:
    def __init__(self, config_path: Path | str = CONFIG_PATH,
                 backend_factory: Callable[[dict], Backend] | None = None):
        self.config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.threshold = float(self.config["confidence_threshold"])
        self._factory = backend_factory or (
            lambda cfg: TransformersBackend(cfg["model_id"], cfg.get("max_length", 64)))
        self._backend: Backend | None = None

    @property
    def backend(self) -> Backend:
        if self._backend is None:
            self._backend = self._factory(self.config)
        return self._backend

    def warm_up(self) -> None:
        """Load the model now (e.g. at container build/start) instead of on the first request."""
        self.classify("مرحبا")

    @property
    def intents(self) -> list[str]:
        return [self.backend.id2label[i] for i in sorted(self.backend.id2label)]

    def classify(self, text: str, top_k: int = 3) -> dict:
        if not text or not text.strip():
            raise ValueError("text must not be empty")
        start = time.perf_counter()
        redacted, pii = redact(text)
        model_input = " ".join(redacted.split())   # same cleaning as training (normalise=False)
        probs = self.backend([model_input])[0]
        order = np.argsort(probs)[::-1][:top_k]
        confidence = float(probs[order[0]])
        return {
            "intent": self.backend.id2label[int(order[0])],
            "confidence": round(confidence, 4),
            "route": "auto" if confidence >= self.threshold else "human",
            "threshold": self.threshold,
            "top_intents": [{"intent": self.backend.id2label[int(i)], "confidence": round(float(probs[i]), 4)}
                            for i in order],
            "redacted_text": model_input,
            "pii_detected": pii,
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        }
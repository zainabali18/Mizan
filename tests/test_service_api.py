"""Service + API tests with a fake model, so CI needs no torch or model download."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mizan.api import create_app
from mizan.service import MizanClassifier

LABELS = {0: "البطاقة المفقودة أو المسروقة", 1: "وصول البطاقة", 2: "حد السن"}


class FakeBackend:
    id2label = LABELS

    def __init__(self, probs):
        self.probs = np.array(probs)
        self.seen = []

    def __call__(self, texts):
        self.seen.extend(texts)
        return np.array([self.probs] * len(texts))


@pytest.fixture
def config(tmp_path):
    p = tmp_path / "mizan_config.json"
    p.write_text(json.dumps({"model_id": "fake/model", "confidence_threshold": 0.85}), encoding="utf-8")
    return p


def make(config, probs):
    backend = FakeBackend(probs)
    return MizanClassifier(config, backend_factory=lambda cfg: backend), backend


def test_confident_prediction_is_auto_routed(config):
    clf, _ = make(config, [0.95, 0.03, 0.02])
    r = clf.classify("بطاقتي انسرقت")
    assert r["intent"] == LABELS[0] and r["route"] == "auto"
    assert [t["intent"] for t in r["top_intents"]] == [LABELS[0], LABELS[1], LABELS[2]]


def test_uncertain_prediction_goes_to_human(config):
    clf, _ = make(config, [0.5, 0.3, 0.2])
    assert clf.classify("مش عارف")["route"] == "human"


def test_model_never_sees_pii(config):
    clf, backend = make(config, [0.9, 0.05, 0.05])
    r = clf.classify("بطاقتي 4111 1111 1111 1111 انسرقت وجوالي 0501234567")
    assert backend.seen == ["بطاقتي [CARD_NUMBER] انسرقت وجوالي [PHONE]"]
    assert r["pii_detected"] == ["CARD_NUMBER", "PHONE"]
    assert "4111" not in json.dumps(r)


def test_empty_text_rejected(config):
    clf, _ = make(config, [1, 0, 0])
    with pytest.raises(ValueError):
        clf.classify("   ")


def test_api_endpoints(config):
    clf, _ = make(config, [0.9, 0.06, 0.04])
    client = TestClient(create_app(clf))
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/intents").json() == list(LABELS.values())

    r = client.post("/classify", json={"text": "بطاقتي 4111 1111 1111 1111 انسرقت"})
    assert r.status_code == 200
    body = r.json()
    assert body["route"] == "auto" and "[CARD_NUMBER]" in body["redacted_text"]
    assert "4111" not in r.text

    assert client.post("/classify", json={"text": ""}).status_code == 422
    assert client.post("/classify", json={"text": "   "}).status_code == 422


def test_validation_errors_never_echo_the_input(config):
    clf, _ = make(config, [0.9, 0.06, 0.04])
    client = TestClient(create_app(clf))
    long_msg = "بطاقتي 4111 1111 1111 1111 " + "ب" * 1000
    for payload in ({"text": long_msg}, {"text": ""}, {}, {"text": 123}):
        r = client.post("/classify", json=payload)
        assert r.status_code == 422
        assert "4111" not in r.text and "input" not in r.text
        assert all({"type", "loc", "msg"} <= set(e) for e in r.json()["detail"])


def test_whitespace_only_error_is_plain(config):
    clf, _ = make(config, [0.9, 0.06, 0.04])
    r = TestClient(create_app(clf)).post("/classify", json={"text": "   "})
    assert r.status_code == 422 and r.json() == {"detail": "text must not be empty"}
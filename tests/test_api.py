"""
End-to-end tests against the FastAPI /predict endpoint.

These tests exercise the full pipeline (model + forensic +
combined verdict) using the FastAPI TestClient — no separate
uvicorn process needed.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    # Imported lazily so the heavy TF model only loads when
    # the API tests actually run.
    from backend.main import app
    return TestClient(app)


def test_home_endpoint(client):

    response = client.get("/")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "success"


def test_predict_returns_combined_verdict_shape(
    client, synthetic_note_bytes
):

    response = client.post(
        "/predict",
        files={
            "file": (
                "note.jpg",
                synthetic_note_bytes,
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "success"

    for key in (
        "prediction",
        "confidence",
        "raw_prediction",
        "model_verdict",
        "model_confidence",
        "forensic_score",
        "forensic_pass_count",
        "forensic_total_checks",
        "forensic_analysis",
    ):
        assert key in body, f"missing key {key} in response"

    assert body["prediction"] in {"REAL", "FAKE", "SUSPICIOUS"}
    assert body["model_verdict"] in {"REAL", "FAKE"}
    assert 0 <= body["forensic_score"] <= 100


def test_predict_handles_invalid_image(client):

    response = client.post(
        "/predict",
        files={
            "file": (
                "not-an-image.txt",
                b"this is plain text",
                "text/plain",
            )
        },
    )

    assert response.status_code == 200
    body = response.json()

    # Either it errored gracefully, or it tried and produced
    # a result — both are acceptable, but never a 500.
    assert body["status"] in {"success", "error"}


def test_predict_blank_image_is_not_real(
    client, blank_image
):

    import cv2

    ok, buf = cv2.imencode(".jpg", blank_image)
    assert ok

    response = client.post(
        "/predict",
        files={
            "file": (
                "blank.jpg",
                buf.tobytes(),
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 200
    body = response.json()

    # A grey square should never be classified as a confident
    # REAL note — forensic checks all fail on it.
    assert body["prediction"] in {"FAKE", "SUSPICIOUS"}

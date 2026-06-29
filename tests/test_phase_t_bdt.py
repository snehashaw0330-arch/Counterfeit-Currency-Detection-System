"""
Phase T2 — BDT counterfeit classifier (backend/bdt_classifier.py).

Model-independent tests: graceful behaviour when untrained, never-raises, and
the 3-band REAL/SUSPICIOUS/FAKE verdict logic exercised with a stub model (so
they pass with or without models/bdt/ present).
"""

import numpy as np
import pytest

import backend.bdt_classifier as bc


_IMG = (np.random.default_rng(0).integers(0, 255, (120, 240, 3))).astype(np.uint8)


@pytest.fixture
def reset_state():
    # snapshot + restore module globals so tests don't leak into each other
    saved = (bc._LOADED, bc._MODEL, bc._SCALER, bc._NAME, bc._REASON, bc._DIR)
    yield
    bc._LOADED, bc._MODEL, bc._SCALER, bc._NAME, bc._REASON, bc._DIR = saved


class _FakeModel:
    classes_ = [0, 1]

    def __init__(self, p_genuine):
        self._p = p_genuine

    def predict_proba(self, x):
        return np.array([[1.0 - self._p, self._p]])


class _IdentityScaler:
    def transform(self, x):
        return x


def test_unavailable_when_untrained(tmp_path, reset_state):
    bc._LOADED, bc._MODEL, bc._SCALER = False, None, None
    bc._DIR = str(tmp_path)  # empty -> no model
    r = bc.predict_bdt(_IMG)
    assert r["available"] is False
    assert r["verdict"] is None


def test_never_raises_on_garbage(reset_state):
    bc._LOADED, bc._MODEL, bc._SCALER, bc._NAME = True, _FakeModel(0.9), _IdentityScaler(), "stub"
    r = bc.predict_bdt(np.zeros((4, 4, 3), np.uint8))
    assert r["available"] in (True, False)  # must not raise


@pytest.mark.parametrize("p,verdict", [(0.85, "REAL"), (0.50, "SUSPICIOUS"), (0.15, "FAKE")])
def test_three_band_verdict(p, verdict, reset_state):
    bc._LOADED, bc._MODEL, bc._SCALER, bc._NAME = True, _FakeModel(p), _IdentityScaler(), "stub"
    r = bc.predict_bdt(_IMG)
    assert r["available"] is True
    assert r["verdict"] == verdict
    assert r["prob_genuine"] == pytest.approx(p, abs=1e-6)

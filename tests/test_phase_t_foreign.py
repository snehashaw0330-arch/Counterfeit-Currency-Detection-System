"""
Per-currency foreign counterfeit classifier (backend/foreign_classifier.py) —
generalises the retired BDT-only bdt_classifier with the same guarantees.

Model-independent tests: graceful behaviour when untrained, never-raises, the
3-band REAL/SUSPICIOUS/FAKE verdict logic with a stub model, and the honesty
flag (synthetic_fakes) propagation — so they pass with or without models/<ccy>/
present on disk.
"""

import numpy as np
import pytest

import backend.foreign_classifier as fc


_IMG = (np.random.default_rng(0).integers(0, 255, (120, 240, 3))).astype(np.uint8)


@pytest.fixture
def clean_cache():
    saved = dict(fc._CACHE)
    fc._CACHE.clear()
    yield
    fc._CACHE.clear()
    fc._CACHE.update(saved)


class _FakeModel:
    classes_ = [0, 1]

    def __init__(self, p_genuine):
        self._p = p_genuine

    def predict_proba(self, x):
        return np.array([[1.0 - self._p, self._p]])


class _IdentityScaler:
    def transform(self, x):
        return x


def _stub(ccy, p_genuine, synthetic=False):
    fc._CACHE[ccy] = {"model": _FakeModel(p_genuine), "scaler": _IdentityScaler(),
                      "name": "stub", "synthetic": synthetic, "reason": None}


def test_unavailable_when_untrained(clean_cache):
    # A currency with no models/<ccy>/ directory must degrade gracefully.
    r = fc.predict_foreign("xyz", _IMG)
    assert r["available"] is False
    assert r["verdict"] is None
    assert "no model" in fc.foreign_model_status("xyz")


def test_never_raises_on_garbage(clean_cache):
    _stub("aud", 0.9)
    r = fc.predict_foreign("aud", np.zeros((4, 4, 3), np.uint8))
    assert r["available"] in (True, False)  # must not raise


@pytest.mark.parametrize("p,verdict", [(0.85, "REAL"), (0.50, "SUSPICIOUS"), (0.15, "FAKE")])
def test_three_band_verdict(p, verdict, clean_cache):
    _stub("bdt", p)
    r = fc.predict_foreign("bdt", _IMG)
    assert r["available"] is True
    assert r["verdict"] == verdict
    assert r["prob_genuine"] == pytest.approx(p, abs=1e-6)
    assert r["currency"] == "BDT"


def test_synthetic_flag_propagates(clean_cache):
    # The honesty caveat must ride along with the verdict so the UI can show it.
    _stub("aud", 0.9, synthetic=True)
    r = fc.predict_foreign("aud", _IMG)
    assert r["available"] is True and r["synthetic_fakes"] is True
    _stub("bdt", 0.9, synthetic=False)
    assert fc.predict_foreign("bdt", _IMG)["synthetic_fakes"] is False


def test_currency_cache_is_per_currency(clean_cache):
    _stub("aud", 0.9)
    _stub("bdt", 0.1)
    assert fc.predict_foreign("aud", _IMG)["verdict"] == "REAL"
    assert fc.predict_foreign("bdt", _IMG)["verdict"] == "FAKE"

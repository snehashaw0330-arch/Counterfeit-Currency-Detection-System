"""Shared fixtures for the test suite."""

import os
import sys

import cv2
import numpy as np
import pytest

# Make the project root importable so `backend.forensic` resolves.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture
def synthetic_note():
    """
    A 600x1200 BGR image that loosely mimics an Indian banknote:
      - light coloured background
      - dark face-like blob on the right
      - vertical security thread in the centre
      - serial-number-like text top-left and bottom-right
      - denomination digits printed in both corners
    """

    img = np.full((600, 1200, 3), (210, 200, 180), dtype=np.uint8)

    # Vertical security thread (thin black line)
    cv2.line(img, (600, 30), (600, 570), (20, 20, 20), 3)

    # "Gandhi face" blob on the right
    cv2.circle(img, (950, 300), 110, (60, 50, 40), -1)
    cv2.circle(img, (920, 280), 10, (250, 250, 250), -1)
    cv2.circle(img, (980, 280), 10, (250, 250, 250), -1)

    # Hologram patch (saturated colour square bottom-left)
    cv2.rectangle(img, (80, 420), (260, 560), (0, 180, 255), -1)
    cv2.rectangle(img, (120, 460), (220, 520), (255, 0, 180), -1)

    # Serial number top-left
    cv2.putText(
        img, "5AB 123456",
        (40, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
        (0, 0, 0), 2, cv2.LINE_AA,
    )

    # Serial number bottom-right
    cv2.putText(
        img, "5AB 123456",
        (820, 560),
        cv2.FONT_HERSHEY_SIMPLEX, 1.4,
        (0, 0, 0), 3, cv2.LINE_AA,
    )

    # Denomination top-right
    cv2.putText(
        img, "500",
        (1020, 90),
        cv2.FONT_HERSHEY_SIMPLEX, 2.2,
        (0, 0, 0), 4, cv2.LINE_AA,
    )

    # Denomination bottom-left
    cv2.putText(
        img, "500",
        (50, 560),
        cv2.FONT_HERSHEY_SIMPLEX, 2.2,
        (0, 0, 0), 4, cv2.LINE_AA,
    )

    return img


@pytest.fixture
def blank_image():
    """A flat grey image — should fail most forensic checks."""

    return np.full((600, 1200, 3), 128, dtype=np.uint8)


@pytest.fixture
def synthetic_note_bytes(synthetic_note):
    """JPEG-encoded bytes of the synthetic note."""

    ok, buf = cv2.imencode(".jpg", synthetic_note)
    assert ok
    return buf.tobytes()

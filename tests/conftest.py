import os
import sys

import pytest

# the same paths scan.py sets up, so tests import exactly what a scan runs
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "starter")]


@pytest.fixture(scope="session")
def det():
    # One Detector for the whole session: building it loads the lexicons and the name model.
    from scanner.detect.text_detector import Detector

    return Detector()


@pytest.fixture
def find(det):
    """find(text, **kw) -> list of (entity, value)."""

    def _f(text, **kw):
        return [(c.entity, c.value) for c in det.detect(text, **kw)]

    return _f

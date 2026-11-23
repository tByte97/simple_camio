"""Auto-detecting a mirrored camera image from the map itself."""

import queue
import threading
from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

from src.core.config_overrides import add_arguments, apply_overrides
from src.config import CameraConfig
from src.core.mirror import MIN_MATCHES, MirrorState

TEMPLATE = Path(__file__).resolve().parents[1] / 'models' / 'SolarSystem' / 'template.png'


def test_fixed_modes_never_ask():
    assert MirrorState('on').flip is True
    assert MirrorState('off').flip is False
    with pytest.raises(ValueError):
        MirrorState('sideways')


def test_apply_flips_only_when_told():
    frame = np.arange(6, dtype=np.uint8).reshape(1, 6)
    assert (MirrorState('off').apply(frame) == frame).all()
    assert (MirrorState('on').apply(frame) == frame[:, ::-1]).all()
    assert (MirrorState('auto').apply(frame) == frame).all()  # undecided: leave as is


@pytest.mark.parametrize('plain, flipped, expected', [
    (150, 6, False),
    (6, 150, True),
    (12, 3, None),                       # too few matches either way: map not in view
    (60, 40, None),                      # no clear winner
])
def test_auto_decides_only_on_a_clear_difference(plain, flipped, expected):
    m = MirrorState('auto')
    assert m.decide(plain, flipped) is (expected is not None)
    assert m.flip is expected


def test_a_decision_sticks():
    m = MirrorState('auto')
    m.decide(6, 150)
    m.decide(150, 6)
    assert m.flip is True


# --- with the real SIFT detector and the SolarSystem map -------------------

@pytest.fixture(scope='module')
def worker():
    if not TEMPLATE.exists():
        pytest.skip('SolarSystem template not present')
    from src.core.workers import SIFTWorker
    from src.detection.sift_detector import SIFTModelDetectorMP
    detector = SIFTModelDetectorMP({'template_image': str(TEMPLATE)})
    return SIFTWorker(detector, queue.Queue(), threading.Lock())


def _scene(flip):
    """The map on a desk: template scaled into a bigger grey frame."""
    template = cv.imread(str(TEMPLATE), cv.IMREAD_GRAYSCALE)
    h, w = template.shape
    scale = 900 / max(h, w)
    small = cv.resize(template, (int(w * scale), int(h * scale)))
    frame = np.full((1080, 1920), 128, np.uint8)
    y, x = (1080 - small.shape[0]) // 2, (1920 - small.shape[1]) // 2
    frame[y:y + small.shape[0], x:x + small.shape[1]] = small
    return cv.flip(frame, 1) if flip else frame


@pytest.mark.parametrize('camera_mirrors', [False, True])
def test_worker_detects_the_camera_orientation(worker, camera_mirrors):
    worker.mirror = MirrorState('auto')
    worker._last_mirror_check_ts = 0.0
    raw = _scene(camera_mirrors)
    plain_n = worker.sift_detector.count_good_matches(raw)
    flipped_n = worker.sift_detector.count_good_matches(cv.flip(raw, 1))
    assert max(plain_n, flipped_n) >= MIN_MATCHES

    seen = worker._settle_mirror(raw)
    assert worker.mirror.flip is camera_mirrors
    assert (seen == _scene(False)).all()   # the worker carries on with the true view


def test_mirror_flag_and_env(monkeypatch):
    import argparse
    monkeypatch.setattr(CameraConfig, 'MIRROR', 'auto')
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    apply_overrides(parser.parse_args(['--mirror', 'off']), environ={})
    assert CameraConfig.MIRROR == 'off'
    apply_overrides(parser.parse_args([]), environ={'CAMIO_MIRROR': 'ON'})
    assert CameraConfig.MIRROR == 'on'
    with pytest.raises(ValueError):
        apply_overrides(parser.parse_args([]), environ={'CAMIO_MIRROR': 'maybe'})

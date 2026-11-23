"""Two fingers held on a zone speak its details; the per-frame signal recorder."""

import csv
import time
from types import SimpleNamespace

import pytest

import simple_camio
from src.config import GestureDetectorConfig
from src.detection.pose_detector import (PoseDetectorMPEnhanced, finger_extension_ratios,
                                         is_two_finger_pose)
from src.detection.signal_recorder import SignalRecorder

HOLD = GestureDetectorConfig.TWO_FINGER_HOLD_SECONDS
GRACE = GestureDetectorConfig.TWO_FINGER_GRACE_SECONDS


def _hand(extended):
    """21 landmarks; fingers named in `extended` lie straight, the rest fold back."""
    points = [(0.5, 0.9)] * 5  # wrist and thumb, irrelevant here
    for column, finger in enumerate(['index', 'middle', 'ring', 'little']):
        x = 0.40 + 0.05 * column
        if finger in extended:
            points += [(x, 0.70), (x, 0.60), (x, 0.50), (x, 0.40)]
        else:  # base, up, then folded back down towards the palm
            points += [(x, 0.70), (x, 0.62), (x + 0.02, 0.66), (x, 0.72)]
    return SimpleNamespace(landmark=[SimpleNamespace(x=x, y=y, z=0.0) for x, y in points])


def test_extension_ratios_tell_straight_from_curled():
    ratios = finger_extension_ratios(_hand({'index', 'middle'}), 1000, 1000)
    assert ratios['index'] == pytest.approx(1.0)
    assert ratios['ring'] < 0.5


@pytest.mark.parametrize('extended, expected', [
    ({'index', 'middle'}, True),
    ({'index'}, False),                                   # ordinary pointing
    ({'index', 'middle', 'ring', 'little'}, False),       # open hand
])
def test_two_finger_pose(extended, expected):
    assert is_two_finger_pose(finger_extension_ratios(_hand(extended), 1000, 1000)) is expected


def test_two_finger_outranks_pointing_but_not_a_double_tap():
    fuse = PoseDetectorMPEnhanced._fuse_detection_outputs
    me = SimpleNamespace(_normalize_index_position=lambda p: p)
    assert fuse(me, None, 'pointing', None, None, 'two_finger', None, False)[1] == 'two_finger'
    assert fuse(me, None, 'double_tap', None, None, 'two_finger', None, False)[1] == 'double_tap'


# --- holding the pose -------------------------------------------------------

HOTSPOTS = {7: {}, 9: {}}


def test_details_come_once_after_the_hold():
    state = {}
    assert not simple_camio.two_finger_details_due(state, 7, HOTSPOTS, 100.0)
    assert not simple_camio.two_finger_details_due(state, 7, HOTSPOTS, 100.0 + HOLD / 2)
    assert simple_camio.two_finger_details_due(state, 7, HOTSPOTS, 100.0 + HOLD)
    assert not simple_camio.two_finger_details_due(state, 7, HOTSPOTS, 100.0 + HOLD + 0.1)


def _frames(state, zone, start, seconds):
    """Call once per 0.1 s frame; return the times at which details fired."""
    steps = round(seconds / 0.1)
    return [round(start + i * 0.1, 2) for i in range(steps + 1)
            if simple_camio.two_finger_details_due(state, zone, HOTSPOTS, start + i * 0.1)]


def test_moving_to_another_zone_starts_a_new_hold():
    state = {}
    assert _frames(state, 7, 100.0, 1.0) == [round(100.0 + HOLD, 2)]
    assert _frames(state, 9, 101.1, 1.0) == [round(101.1 + HOLD, 2)]


def test_the_background_never_plays():
    state = {}
    simple_camio.two_finger_details_due(state, 16777215, HOTSPOTS, 100.0)
    assert not simple_camio.two_finger_details_due(state, 16777215, HOTSPOTS, 100.0 + HOLD)


def test_a_brief_flicker_keeps_the_hold_but_a_real_break_resets_it():
    state = {}
    simple_camio.two_finger_details_due(state, 7, HOTSPOTS, 100.0)
    simple_camio.two_finger_details_due(state, 7, HOTSPOTS, 100.0 + GRACE * 0.8)
    assert simple_camio.two_finger_details_due(state, 7, HOTSPOTS, 100.0 + HOLD)

    state = {}
    simple_camio.two_finger_details_due(state, 7, HOTSPOTS, 100.0)
    assert not simple_camio.two_finger_details_due(state, 7, HOTSPOTS, 100.0 + GRACE + 0.05)


def test_the_main_loop_asks_for_details_like_a_double_tap():
    commands = []
    worker = SimpleNamespace(enqueue_command=commands.append)
    components = SimpleNamespace(
        interact=SimpleNamespace(push_gesture=lambda loc: 7),
        camio_player=SimpleNamespace(hotspots={7: {'textDescription': 'Марс'}}, prev_zone_name=''),
    )
    state = {'was_detected': True, 'description_played': True, 'first_detected_ts': 1.0,
             'missing_since': 0.0,
             'two_finger': {'zone': 7, 'since': time.time() - HOLD - 0.1, 'seen': time.time(), 'played': False}}
    simple_camio.process_gestures_and_audio([1.0, 1.0, 0.0], 'two_finger', components, 0.0, worker, state)
    assert [(c.command_type, c.params) for c in commands] == [
        ('play_zone', {'zone_id': 7, 'gesture_status': 'double_tap'})]


# --- signal recorder --------------------------------------------------------


def test_recorder_writes_a_row_per_hand_with_tap_events(tmp_path):
    path = tmp_path / 'signals.csv'
    recorder = SignalRecorder(str(path))
    base = SimpleNamespace(_tap_state={'Right': {'pressing': False, 'event': 'tap'}},
                           _detect_pointing_gesture=lambda lm, w, h: True)
    enh = SimpleNamespace(
        _tap_state_plus={'Right': {'pressing': False, 'gates': (False, False, 1)}},
        _extract_hand_metadata=lambda results, i: ('Right', 0.97),
        _compute_enhanced_signals=lambda lm, w, h: {'angle_deg': 5.0, 'zrel': -0.01, 'plane_d': 0.02},
    )
    results = SimpleNamespace(multi_hand_landmarks=[_hand({'index'})])
    recorder.record(base, enh, (results, 1920, 1080), 'pointing')
    recorder.record(base, enh, (results, 1920, 1080), 'pointing')
    recorder.close()

    rows = list(csv.DictReader(path.open(encoding='utf-8')))
    assert [r['base_event'] for r in rows] == ['tap', '']   # an event is reported once
    assert rows[0]['hand_key'] == 'Right' and rows[0]['two_finger'] == '0'
    assert rows[0]['enh_stable'] == '0' and rows[0]['enh_triggers'] == '1'

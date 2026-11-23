"""
Why a zone stays silent when it should speak.

A 6-minute session on the Pi logged 136 tracker resets - one per dropout where
MediaPipe briefly lost the hand, 97 of them under 0.1 s - and 116
announcements suppressed as `same_name` against 76 that played. In the 0.8-2s
window right after a reset the ratio was 39 suppressed to 10 played. Both
halves are covered here: the main loop no longer treats a momentary dropout as
the hand leaving, and a double-tap no longer loses to the change check.
"""

import time

import simple_camio
from src.audio.audio import ZoneAudioPlayer
from src.config import InteractionConfig

GRACE = InteractionConfig.HAND_LOSS_GRACE_SECONDS


class FakeAudioWorker:
    """Records commands instead of playing them."""

    def __init__(self):
        self.commands = []

    def enqueue_command(self, command):
        self.commands.append(command.command_type)


def _hand_state(missing_for=None):
    """A tracked hand, optionally already missing for that many seconds."""
    return {
        'was_detected': True,
        'description_played': True,
        'first_detected_ts': 100.0,
        'missing_since': 0.0 if missing_for is None else time.time() - missing_for,
    }


# --- one dropped frame is not the hand leaving ----------------------------


def test_a_momentary_dropout_keeps_the_hand():
    """
    The pose worker republishes ~14 times a second while the loop polls at ~30,
    so a dropped detection is routine - the median one lasted 3 ms. Acting on
    it restarted the whole "hand appeared" scenario and marked the zone under
    the finger as already spoken.
    """
    state = _hand_state()
    worker = FakeAudioWorker()

    simple_camio.mark_hand_missing(state, worker)
    simple_camio.mark_hand_missing(state, worker)

    assert state['was_detected'] is True
    assert state['first_detected_ts'] == 100.0
    assert worker.commands == []


def test_the_first_missing_frame_starts_the_clock():
    """Without a start time the window could never expire."""
    state = _hand_state()

    simple_camio.mark_hand_missing(state, FakeAudioWorker())

    assert state['missing_since'] > 0


def test_the_hand_still_leaves_once_the_grace_runs_out():
    """The other side of the gate, so the tests above cannot pass vacuously."""
    state = _hand_state(missing_for=GRACE + 0.1)
    worker = FakeAudioWorker()

    simple_camio.mark_hand_missing(state, worker)

    assert state['was_detected'] is False
    assert state['first_detected_ts'] == 0.0
    assert worker.commands == ['heartbeat_pause', 'crickets_play']


def test_a_recovered_frame_clears_the_clock():
    """
    Flicker is a near-miss, a hit, then another near-miss. Without the reset
    the second one would inherit the first one's start time and tear down a
    hand that never actually left.
    """
    state = _hand_state(missing_for=GRACE - 0.05)
    worker = FakeAudioWorker()

    state['missing_since'] = 0.0  # what a valid gesture does
    simple_camio.mark_hand_missing(state, worker)

    assert state['was_detected'] is True
    assert worker.commands == []


def test_crickets_start_once_not_every_frame_the_hand_is_gone():
    """A hand off the map for a minute must not queue a command per frame."""
    state = _hand_state(missing_for=GRACE + 0.1)
    worker = FakeAudioWorker()

    for _ in range(30):
        simple_camio.mark_hand_missing(state, worker)

    assert worker.commands == ['heartbeat_pause', 'crickets_play']


# --- double-tap is an explicit repeat --------------------------------------


def _player():
    """A player with the zone bookkeeping convey() touches, and nothing else."""
    player = ZoneAudioPlayer.__new__(ZoneAudioPlayer)
    player.hotspots = {7: {'textDescription': 'Бухгалтерія'}}
    player.prev_zone_name = 'Бухгалтерія'
    player.played = []
    player._play_zone_audio = player.played.append
    player.detail_files = {}
    player._play_zone_detail = lambda zone: player.played.append(('detail', zone))
    return player


def test_a_double_tap_repeats_the_zone_it_just_announced():
    """
    Of 6 double-taps in the Pi session exactly one made a sound; the rest lost
    to this check or landed on the background.
    """
    player = _player()

    player.convey(7, 'double_tap')

    assert player.played == [7]


def test_pointing_at_the_same_zone_still_says_nothing():
    """Otherwise a resting finger would repeat its zone forever."""
    player = _player()

    player.convey(7, 'pointing')

    assert player.played == []


def test_a_double_tap_on_the_background_still_says_nothing():
    """The zone has to exist; 4 of the 6 logged double-taps hit the backdrop."""
    player = _player()

    player.convey(16777215, 'double_tap')

    assert player.played == []
    assert player.prev_zone_name is None


# --- a double-tap speaks the zone's detail sentence when it has one --------


def test_a_double_tap_plays_the_detail_sentence_when_the_zone_has_one():
    player = _player()
    player.detail_files = {7: object()}

    player.convey(7, 'double_tap')

    assert player.played == [('detail', 7)]


def test_pointing_never_plays_the_detail_sentence():
    player = _player()
    player.detail_files = {7: object()}
    player.prev_zone_name = None

    player.convey(7, 'pointing')

    assert player.played == [7]


# --- a double-tap that drifts off a small zone still means that zone -------


class RecordingWorker:
    def __init__(self):
        self.commands = []

    def enqueue_command(self, command):
        self.commands.append(command)


def _components(zones_under_finger):
    from types import SimpleNamespace
    zones = iter(zones_under_finger)
    return SimpleNamespace(
        interact=SimpleNamespace(push_gesture=lambda loc: next(zones)),
        camio_player=SimpleNamespace(hotspots={7: {'textDescription': 'Меркурій'}},
                                     prev_zone_name='Меркурій'),
    )


def _double_tap_zone(hand_state, components):
    worker = RecordingWorker()
    simple_camio.process_gestures_and_audio(
        [10.0, 10.0, 0.0], 'double_tap', components, 0.0, worker, hand_state)
    return [c.params.get('zone_id') for c in worker.commands if c.command_type == 'play_zone']


def test_a_double_tap_on_the_background_uses_the_zone_the_finger_just_left():
    """A 1 cm planet is easy to miss: the logged double tap landed on 0xFFFFFF."""
    state = _hand_state()
    components = _components([7, 16777215])
    worker = RecordingWorker()
    simple_camio.process_gestures_and_audio([1.0, 1.0, 0.0], 'pointing', components, 0.0, worker, state)

    assert _double_tap_zone(state, components) == [7]


def test_an_old_zone_is_not_reused_by_a_double_tap():
    state = _hand_state()
    state['last_zone'] = (7, time.time() - InteractionConfig.DOUBLE_TAP_ZONE_MEMORY_SECONDS - 1)

    assert _double_tap_zone(state, _components([16777215])) == [16777215]


def test_a_hand_that_left_forgets_its_zone():
    state = _hand_state(missing_for=GRACE + 0.1)
    state['last_zone'] = (7, time.time())

    simple_camio.mark_hand_missing(state, FakeAudioWorker())

    assert 'last_zone' not in state

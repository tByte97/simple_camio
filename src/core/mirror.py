"""
Camera mirroring.

Some webcams (and some driver settings) deliver a mirrored, selfie-style
image, others the true view. SIFT descriptors are not mirror-invariant, so a
mirrored frame barely matches the map template and the map is never found.

In 'auto' mode the SIFT worker compares how many template features match the
frame as captured and flipped left-right, and settles the question once the
difference is clear. From then on every frame is flipped at capture, before
SIFT, MediaPipe and the display see it, so all of them agree.

Modes: 'auto' (default), 'on' (always flip), 'off' (never flip).
"""

import logging

import cv2 as cv

logger = logging.getLogger(__name__)

MODES = ('auto', 'on', 'off')

# The winning orientation needs at least this many good matches...
MIN_MATCHES = 20
# ...and this many times as many as the other orientation.
MIN_RATIO = 2.0


class MirrorState:
    """Whether captured frames are flipped. `flip` is None until auto mode decides."""

    def __init__(self, mode='auto'):
        if mode not in MODES:
            raise ValueError(f'mirror mode must be one of {MODES}, got {mode!r}')
        self.mode = mode
        self.flip = {'on': True, 'off': False}.get(mode)

    @property
    def decided(self):
        return self.flip is not None

    def apply(self, frame):
        """Return the frame as the rest of the system should see it."""
        return cv.flip(frame, 1) if self.flip else frame

    def decide(self, plain_matches, flipped_matches):
        """
        Settle auto mode from match counts of the raw and the flipped frame.

        Returns True once decided (now or earlier), False if still unclear.
        """
        if self.decided:
            return True
        best, other = max(plain_matches, flipped_matches), min(plain_matches, flipped_matches)
        if best < MIN_MATCHES or best < MIN_RATIO * other:
            return False
        self.flip = flipped_matches > plain_matches
        if self.flip:
            logger.info(f"Camera image is mirrored (map matches: as captured {plain_matches}, "
                        f"flipped {flipped_matches}); flipping every frame. Override with --mirror off.")
        else:
            logger.info(f"Camera image is not mirrored (map matches: as captured {plain_matches}, "
                        f"flipped {flipped_matches}). Override with --mirror on.")
        return True

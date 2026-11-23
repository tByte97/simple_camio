"""
Per-frame CSV of the finger signals tap detection works from.

Turned on with --record-taps FILE. One row per detected hand per pose frame,
including the events both tap detectors produced on that frame, so a recorded
session of known double taps shows what the detectors actually saw.
"""

import csv
import logging
import time

from src.detection.pose_detector import finger_extension_ratios, is_two_finger_pose

logger = logging.getLogger(__name__)

COLUMNS = [
    't', 'frame', 'hand_idx', 'hand_key', 'score',
    'tip_x', 'tip_y', 'tip_z', 'angle_deg', 'zrel', 'plane_d',
    'r_index', 'r_middle', 'r_ring', 'r_little', 'pointing', 'two_finger',
    'base_pressing', 'base_event', 'enh_pressing', 'enh_event',
    'enh_stable', 'enh_conf_ok', 'enh_triggers', 'status',
]


class SignalRecorder:
    """Appends one CSV row per hand per pose frame; line-buffered so a crash keeps the data."""

    def __init__(self, path):
        self.path = path
        self._file = open(path, 'w', newline='', encoding='utf-8', buffering=1)
        self._writer = csv.DictWriter(self._file, fieldnames=COLUMNS)
        self._writer.writeheader()
        self._frame = 0
        logger.info(f"Recording finger signals to {path}")

    def record(self, base, enh, mp_results, status):
        """Write rows for every hand in this frame's MediaPipe results."""
        results, width, height = mp_results
        self._frame += 1
        if not (results and results.multi_hand_landmarks):
            return
        now = time.time()
        for idx, landmarks in enumerate(results.multi_hand_landmarks):
            hand_key, score = enh._extract_hand_metadata(results, idx)
            signals = enh._compute_enhanced_signals(landmarks, width, height)
            ratios = finger_extension_ratios(landmarks, width, height)
            tip = landmarks.landmark[8]
            b = base._tap_state.get(hand_key, {})
            e = enh._tap_state_plus.get(hand_key, {})
            stable, conf_ok, triggers = e.get('gates', ('', '', ''))
            self._writer.writerow({
                't': f'{now:.3f}', 'frame': self._frame, 'hand_idx': idx,
                'hand_key': hand_key, 'score': f'{score:.3f}' if score is not None else '',
                'tip_x': f'{tip.x * width:.1f}', 'tip_y': f'{tip.y * height:.1f}',
                'tip_z': f'{tip.z:.5f}', 'angle_deg': f"{signals['angle_deg']:.2f}",
                'zrel': f"{signals['zrel']:.5f}", 'plane_d': f"{signals['plane_d']:.5f}",
                'r_index': f"{ratios['index']:.3f}", 'r_middle': f"{ratios['middle']:.3f}",
                'r_ring': f"{ratios['ring']:.3f}", 'r_little': f"{ratios['little']:.3f}",
                'pointing': int(base._detect_pointing_gesture(landmarks, width, height)),
                'two_finger': int(is_two_finger_pose(ratios)),
                'base_pressing': int(bool(b.get('pressing'))), 'base_event': b.pop('event', ''),
                'enh_pressing': int(bool(e.get('pressing'))), 'enh_event': e.pop('event', ''),
                'enh_stable': int(stable) if stable != '' else '',
                'enh_conf_ok': int(conf_ok) if conf_ok != '' else '',
                'enh_triggers': triggers,
                'status': status or '',
            })

    def close(self):
        self._file.close()

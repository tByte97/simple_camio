"""
Runtime configuration overrides.

The config classes hold their values as class attributes, so changing one used
to mean editing source - README still tells the reader to set
TapDetectionConfig.COLLECT_TAP_DATA by hand. These helpers let a flag or an
environment variable do it instead, extending the pattern --headless already
established.

Precedence: command-line flag, then CAMIO_* environment variable, then the
value on the class.
"""

import ast
import logging
import os

import cv2 as cv

from src import config
from src.config import CameraConfig, TapDetectionConfig
from src.core.mirror import MODES as MIRROR_MODES

logger = logging.getLogger(__name__)

# Every name must map to a real OpenCV constant. 'auto' used to map to None,
# which the camera setup read as "no preference" and replaced with DirectShow -
# so on Linux it silently broke the camera. 'any' is cv.CAP_ANY, OpenCV's own
# auto-selection, and covers that intent.
BACKENDS = {
    'v4l2': cv.CAP_V4L2,
    'dshow': cv.CAP_DSHOW,
    'msmf': cv.CAP_MSMF,
    'any': cv.CAP_ANY,
}

LOG_LEVELS = ('DEBUG', 'INFO', 'WARNING', 'ERROR')

TRUTHY = ('1', 'true', 'yes', 'on')


def _resolution(value, source):
    """Parse WxH into a pair of ints, naming its source if it is malformed."""
    try:
        width, height = value.lower().split('x')
        return int(width), int(height)
    except (ValueError, AttributeError):
        raise ValueError(
            f'{source}: expected a resolution like 1280x720, got {value!r}'
        )


def add_arguments(parser):
    """Register the override flags on an existing parser."""
    parser.add_argument(
        '--headless', action='store_true', default=False,
        help='Run without a display window - useful for Raspberry Pi daemon mode'
    )
    parser.add_argument(
        '--resolution', type=lambda v: _resolution(v, '--resolution'), default=None,
        metavar='WxH', help='Camera capture resolution, e.g. 640x480'
    )
    parser.add_argument(
        '--camera-backend', choices=sorted(BACKENDS), default=None,
        help='OpenCV capture backend. v4l2 on Linux, dshow or msmf on Windows'
    )
    parser.add_argument(
        '--collect-tap-data', action='store_true', default=False,
        help='Record tap detection samples to data/tap_dataset for training'
    )
    parser.add_argument(
        '--log-level', choices=LOG_LEVELS, default=None,
        help='Logging verbosity (default: INFO)'
    )
    parser.add_argument(
        '--mirror', choices=MIRROR_MODES, default=None,
        help="Flip the camera image left-right: auto (default) detects it from "
             "the map, on always flips, off never does"
    )
    parser.add_argument(
        '--record-taps', default=None, metavar='FILE.csv',
        help='Write per-frame finger signals and tap events to a CSV for tuning'
    )
    parser.add_argument(
        '--set', action='append', default=[], metavar='CLASS.NAME=VALUE',
        help='Override any value in src/config.py for this run, e.g. '
             '--set TapDetectionConfig.TAP_MAX_INTERVAL=1.5 (repeatable)'
    )


def parse_setting(text):
    """
    Turn 'TapDetectionConfig.TAP_MAX_INTERVAL=1.5' into (class, name, value).

    Only existing attributes of the *Config classes can be set, and the value
    must keep the attribute's type (an int may become a float), so a typo
    fails at startup instead of silently doing nothing.
    """
    target, sep, raw = text.partition('=')
    cls_name, dot, name = target.strip().partition('.')
    cls = getattr(config, cls_name, None)
    if not (sep and dot and cls_name.endswith('Config') and isinstance(cls, type)):
        raise ValueError(f'--set {text!r}: expected Config class like TapDetectionConfig.NAME=VALUE')
    if not name.isupper() or not hasattr(cls, name):
        raise ValueError(f'--set {text!r}: {cls_name} has no setting {name}')
    current = getattr(cls, name)
    try:
        value = ast.literal_eval(raw.strip())
    except (ValueError, SyntaxError):
        raise ValueError(f'--set {text!r}: cannot parse value {raw!r}') from None
    numeric = (int, float)
    if isinstance(current, bool) or isinstance(value, bool):
        ok = isinstance(current, bool) and isinstance(value, bool)
    elif isinstance(current, numeric):
        ok = isinstance(value, numeric)
        value = float(value) if isinstance(current, float) else value
    else:
        ok = current is None or isinstance(value, type(current))
    if not ok:
        raise ValueError(f'--set {text!r}: {name} is {type(current).__name__}, got {value!r}')
    return cls, name, value


def apply_overrides(args, environ=None):
    """
    Apply command-line and environment overrides to the config classes.

    Args:
        args (argparse.Namespace): Parsed arguments from a parser that
            add_arguments() was called on.
        environ (dict, optional): Environment to read. Defaults to os.environ.

    Returns:
        list[str]: One line per override applied, for logging.

    Raises:
        ValueError: If a CAMIO_* variable holds a value that cannot be parsed.
    """
    env = os.environ if environ is None else environ
    applied = []

    resolution = getattr(args, 'resolution', None)
    if resolution is None and env.get('CAMIO_RESOLUTION'):
        resolution = _resolution(env['CAMIO_RESOLUTION'], 'CAMIO_RESOLUTION')
    if resolution is not None:
        CameraConfig.DEFAULT_WIDTH, CameraConfig.DEFAULT_HEIGHT = resolution
        applied.append(f'CameraConfig.DEFAULT_WIDTH/HEIGHT = {resolution[0]}x{resolution[1]}')

    backend = getattr(args, 'camera_backend', None)
    if backend is None and env.get('CAMIO_CAMERA_BACKEND'):
        backend = env['CAMIO_CAMERA_BACKEND'].lower()
        if backend not in BACKENDS:
            raise ValueError(
                f'CAMIO_CAMERA_BACKEND: expected one of {sorted(BACKENDS)}, '
                f'got {backend!r}'
            )
    if backend is not None:
        CameraConfig.BACKEND = BACKENDS[backend]
        applied.append(f'CameraConfig.BACKEND = {backend}')

    headless = getattr(args, 'headless', False)
    if not headless:
        headless = env.get('CAMIO_HEADLESS', '').lower() in TRUTHY
    if headless:
        CameraConfig.HEADLESS = True
        applied.append('CameraConfig.HEADLESS = True')

    collect = getattr(args, 'collect_tap_data', False)
    if not collect:
        collect = env.get('CAMIO_COLLECT_TAP_DATA', '').lower() in TRUTHY
    if collect:
        TapDetectionConfig.COLLECT_TAP_DATA = True
        applied.append('TapDetectionConfig.COLLECT_TAP_DATA = True')

    level = getattr(args, 'log_level', None)
    if level is None and env.get('CAMIO_LOG_LEVEL'):
        level = env['CAMIO_LOG_LEVEL'].upper()
        if level not in LOG_LEVELS:
            raise ValueError(
                f'CAMIO_LOG_LEVEL: expected one of {list(LOG_LEVELS)}, got {level!r}'
            )
    if level is not None:
        logging.getLogger().setLevel(getattr(logging, level))
        applied.append(f'log level = {level}')

    mirror = getattr(args, 'mirror', None)
    if mirror is None and env.get('CAMIO_MIRROR'):
        mirror = env['CAMIO_MIRROR'].lower()
        if mirror not in MIRROR_MODES:
            raise ValueError(
                f'CAMIO_MIRROR: expected one of {list(MIRROR_MODES)}, got {mirror!r}'
            )
    if mirror is not None:
        CameraConfig.MIRROR = mirror
        applied.append(f'CameraConfig.MIRROR = {mirror}')

    record = getattr(args, 'record_taps', None)
    if record:
        TapDetectionConfig.RECORD_SIGNALS_PATH = record
        applied.append(f'TapDetectionConfig.RECORD_SIGNALS_PATH = {record}')

    # Parse every --set first so one bad entry changes nothing.
    settings = [parse_setting(text) for text in getattr(args, 'set', None) or []]
    for cls, name, value in settings:
        setattr(cls, name, value)
        applied.append(f'{cls.__name__}.{name} = {value!r}')

    return applied

"""Project-wide logging: one console handler on the "salnav" root logger.

Usage:  log = get_logger(__name__)
"""

import logging
import sys

_ROOT_NAME = "salnav"
_FORMAT = "[%(asctime)s] %(levelname)-7s %(name)s: %(message)s"
_DATEFMT = "%H:%M:%S"


def get_logger(name: str = None) -> logging.Logger:
    root = logging.getLogger(_ROOT_NAME)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
    if not name or name == _ROOT_NAME:
        return root
    short = name.rsplit(".", 1)[-1]
    return root.getChild(short)

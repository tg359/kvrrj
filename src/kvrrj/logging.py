"""Logging utilities."""

import logging
import sys

# logging.captureWarnings(True)
# todo - fix warnings not being shown in console through CONSOLE_LOGGER
# get current module name
TOOLKIT_NAME = __name__.split(".")[0]

formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)

CONSOLE_LOGGER = logging.getLogger(f"{TOOLKIT_NAME}[console]")
CONSOLE_LOGGER.propagate = False
CONSOLE_LOGGER.setLevel(logging.DEBUG)
CONSOLE_LOGGER.addHandler(handler)

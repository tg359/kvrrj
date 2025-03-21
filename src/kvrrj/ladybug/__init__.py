"""Base module for the ladybugtools_toolkit package."""

# pylint: disable=E0401
import getpass
import os
from pathlib import Path

import matplotlib.pyplot as plt

# pylint: disable=E0401


# get common paths
DATA_DIRECTORY = (Path(__file__).parent.parent / "data").absolute()
HOME_DIRECTORY = (Path("C:/Users/") / getpass.getuser()).absolute()

# override "HOME" in case IT has set this to something other than default
os.environ["HOME"] = HOME_DIRECTORY.as_posix()

# set plotting style for modules within this toolkit
plt.style.use(DATA_DIRECTORY / "bhom.mplstyle")

from pathlib import Path

from ladybug.epw import EPW

EPW_OBJ = EPW(Path(__file__).parent / "assets" / "example.epw")

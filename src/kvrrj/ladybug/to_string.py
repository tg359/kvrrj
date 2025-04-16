"""Methods for converting various objects into custom string representations of themselves."""

from functools import singledispatch
from typing import Any

from ladybug.analysisperiod import AnalysisPeriod
from ladybug.location import Location

from .util import _analysis_period_to_string


@singledispatch
def to_string(obj: Any, **kwargs) -> str:
    """Convert an object to a string."""
    raise NotImplementedError(f"Cannot convert {type(obj)} to a custom string.")


@to_string.register(AnalysisPeriod)
def _(obj: AnalysisPeriod, **kwargs) -> str:
    return _analysis_period_to_string(obj, **kwargs)


@to_string.register(Location)
def _(obj: Location, **kwargs) -> str:
    return f"{obj.country.strip()} - {obj.city.strip()}"

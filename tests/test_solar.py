import pandas as pd
import pytest
from ladybug.location import Location

from kvrrj.solar import Solar

from . import EPW_OBJ


def test_init():
    location = Location()
    datetimes = (
        pd.date_range(start="2023-01-01", periods=24, freq="h", tz="UTC")
        .to_pydatetime()
        .tolist()
    )
    dni = [100.0] * 24
    dhi = [50.0] * 24
    ghi = [150.0] * 24

    with pytest.warns(UserWarning):
        solar = Solar(
            location=location,
            datetimes=datetimes,
            direct_normal_radiation=dni,
            diffuse_horizontal_radiation=dhi,
            global_horizontal_radiation=ghi,
        )

    assert solar.location == location
    assert solar.datetimes == datetimes
    assert solar.direct_normal_radiation == dni
    assert solar.diffuse_horizontal_radiation == dhi
    assert solar.global_horizontal_radiation == ghi


def test_from_epw():
    assert isinstance(Solar.from_epw(EPW_OBJ), Solar)


def test_len():
    obj = Solar.from_epw(EPW_OBJ)
    len(obj)


def test_str():
    obj = Solar.from_epw(EPW_OBJ)
    str(obj)


def test_repr():
    obj = Solar.from_epw(EPW_OBJ)
    repr(obj)


def test_hash():
    obj = Solar.from_epw(EPW_OBJ)
    hash(obj)


def test_eq():
    obj = Solar.from_epw(EPW_OBJ)
    obj == obj


def test_iter():
    obj = Solar.from_epw(EPW_OBJ)
    for i in obj:
        pass


def test_getitem():
    obj = Solar.from_epw(EPW_OBJ)
    obj[2]


def test_copy():
    obj = Solar.from_epw(EPW_OBJ)
    obj.__copy__()


def test_from_pvlib():
    loc = Location()
    obj = Solar.from_pvlib(location=loc, start_date="2017-01-01", end_date="2017-12-31")
    assert isinstance(obj, Solar)


def test_to_dict():
    obj = Solar.from_epw(EPW_OBJ)
    obj_dict = obj.to_dict()
    assert isinstance(obj_dict, dict)


def test_from_dict():
    obj = Solar.from_epw(EPW_OBJ)
    obj_dict = obj.to_dict()
    new_obj = Solar.from_dict(obj_dict)
    assert isinstance(new_obj, Solar)


def test_to_json():
    obj = Solar.from_epw(EPW_OBJ)
    obj_json = obj.to_json()
    assert isinstance(obj_json, str)


def test_from_json():
    obj = Solar.from_epw(EPW_OBJ)
    obj_json = obj.to_json()
    new_obj = Solar.from_json(obj_json)
    assert isinstance(new_obj, Solar)


def test_from_dataframe():
    obj = Solar.from_epw(EPW_OBJ)
    obj_df = obj.df
    new_obj = Solar.from_dataframe(obj_df, location=obj.location)
    assert isinstance(new_obj, Solar)


def test_from_average():
    obj = Solar.from_epw(EPW_OBJ)
    new_obj = Solar.from_average([obj, obj, obj])
    assert isinstance(new_obj, Solar)


def test_dates():
    solar = Solar.from_epw(EPW_OBJ)
    solar.dates


def test_start_date():
    solar = Solar.from_epw(EPW_OBJ)
    solar.start_date


def test_end_date():
    solar = Solar.from_epw(EPW_OBJ)
    solar.end_date


def test_lb_datetimes():
    solar = Solar.from_epw(EPW_OBJ)
    solar.lb_datetimes


def test_lb_dates():
    solar = Solar.from_epw(EPW_OBJ)
    solar.lb_dates


def test_datetimeindex():
    solar = Solar.from_epw(EPW_OBJ)
    solar.datetimeindex


def test_direct_normal_radiation_series():
    solar = Solar.from_epw(EPW_OBJ)
    solar.direct_normal_radiation_series


def test_direct_normal_radiation_collection():
    solar = Solar.from_epw(EPW_OBJ)
    solar.direct_normal_radiation_collection


def test_diffuse_horizontal_radiation_series():
    solar = Solar.from_epw(EPW_OBJ)
    solar.diffuse_horizontal_radiation_series


def test_diffuse_horizontal_radiation_collection():
    solar = Solar.from_epw(EPW_OBJ)
    solar.diffuse_horizontal_radiation_collection


def test_global_horizontal_radiation_series():
    solar = Solar.from_epw(EPW_OBJ)
    solar.global_horizontal_radiation_series


def test_global_horizontal_radiation_collection():
    solar = Solar.from_epw(EPW_OBJ)
    solar.global_horizontal_radiation_collection


def test_df():
    solar = Solar.from_epw(EPW_OBJ)
    solar.df


def test_analysis_period():
    solar = Solar.from_epw(EPW_OBJ)
    solar.analysis_period


def test_sunpath():
    solar = Solar.from_epw(EPW_OBJ)
    solar.sunpath


def test_suns():
    solar = Solar.from_epw(EPW_OBJ)
    solar.suns


def test_suns_df():
    solar = Solar.from_epw(EPW_OBJ)
    solar.suns_df


def test_sunrise_sunset():
    solar = Solar.from_epw(EPW_OBJ)
    solar.sunrise_sunset


def test_solstices_equinoxes():
    solar = Solar.from_epw(EPW_OBJ)
    solar.solstices_equinoxes


def test_to_wea():
    solar = Solar.from_epw(EPW_OBJ)
    solar.to_wea()


def test_apply_shade_objects():
    solar = Solar.from_epw(EPW_OBJ)
    with pytest.raises(NotImplementedError):
        solar.apply_shade_objects()


def test__sky_matrix():
    solar = Solar.from_epw(EPW_OBJ)
    solar._sky_matrix()


def test__radiation_rose():
    solar = Solar.from_epw(EPW_OBJ)
    solar._radiation_rose()


def test__radiation_benefit_data():
    solar = Solar.from_epw(EPW_OBJ)
    solar._radiation_benefit_data(temperature=EPW_OBJ.dry_bulb_temperature)


def test__radiation_rose_data():
    solar = Solar.from_epw(EPW_OBJ)
    solar._radiation_rose_data()


def test__tilt_orientation_factor_data():
    solar = Solar.from_epw(EPW_OBJ)
    solar._tilt_orientation_factor_data()


def test_filter_by_boolean_mask():
    solar = Solar.from_epw(EPW_OBJ)
    solar.filter_by_boolean_mask()


def test_filter_by_analysis_period():
    solar = Solar.from_epw(EPW_OBJ)
    solar.filter_by_analysis_period()


def test_filter_by_time():
    solar = Solar.from_epw(EPW_OBJ)
    solar.filter_by_time()


def test_plot_radiation_rose():
    solar = Solar.from_epw(EPW_OBJ)
    solar.plot_radiation_rose()


def test_plot_tilt_orientation_factor():
    solar = Solar.from_epw(EPW_OBJ)
    solar.plot_tilt_orientation_factor()


def test_plot_radiation_benefit_heatmap():
    solar = Solar.from_epw(EPW_OBJ)
    solar.plot_radiation_benefit_heatmap(temperature=EPW_OBJ.dry_bulb_temperature)


def test_plot_sunpath():
    solar = Solar.from_epw(EPW_OBJ)
    solar.plot_sunpath()


def test_plot_skymatrix():
    solar = Solar.from_epw(EPW_OBJ)
    solar.plot_skymatrix()


def test_plot_hours_sunlight():
    solar = Solar.from_epw(EPW_OBJ)
    solar.plot_hours_sunlight()


def test_plot_solar_elevation_azimuth():
    solar = Solar.from_epw(EPW_OBJ)
    solar.plot_solar_elevation_azimuth()

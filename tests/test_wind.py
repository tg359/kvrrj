import numpy as np
import pandas as pd
from ladybug.analysisperiod import AnalysisPeriod

from kvrrj.wind import Location, TerrainType, Wind

from . import EPW_OBJ

# region: GENERAL


def test_terrain_type():
    for i in TerrainType:
        assert isinstance(i, TerrainType)
        i.boundary_layer_height
        i.power_law_exponent
        i.roughness_length

    TerrainType.from_roughness_length(0.15)

    for tt in TerrainType:
        for rv in [0, 0.01, 5, 20]:
            for rh in [0.1, 1, 10, 100]:
                for th in [0, 0.1, 10]:
                    for ll in [True, False]:
                        tt.translate_wind_speed(
                            reference_value=rv,
                            reference_height=rh,
                            target_height=th,
                            log_law=ll,
                            target_terrain_type=tt,
                        )


# endregion: GENERAL

# region: INIT


def test_init():
    location = Location()
    datetimes = (
        pd.date_range(start="2017-01-01", periods=8760, freq="h", tz="UTC")
        .to_pydatetime()
        .tolist()
    )
    ws = [1] * 8760
    wd = [90] * 8760
    obj = Wind(
        location=location,
        datetimes=datetimes,
        wind_speed=ws,
        wind_direction=wd,
        height_above_ground=10,
        terrain_type=TerrainType.COUNTRY,
    )

    assert isinstance(obj, Wind)


# endregion: INIT

# region: CLASS METHODS


def test_from_epw():
    obj = Wind.from_epw(EPW_OBJ)
    assert isinstance(obj, Wind)


def test_from_openmeteo():
    loc = Location()
    obj = Wind.from_openmeteo(
        location=loc, start_date="2017-01-01", end_date="2017-12-31"
    )
    assert isinstance(obj, Wind)


def test_to_dict():
    obj = Wind.from_epw(EPW_OBJ)
    obj_dict = obj.to_dict()
    assert isinstance(obj_dict, dict)


def test_from_dict():
    obj = Wind.from_epw(EPW_OBJ)
    obj_dict = obj.to_dict()
    new_obj = Wind.from_dict(obj_dict)
    assert isinstance(new_obj, Wind)


def test_to_json():
    obj = Wind.from_epw(EPW_OBJ)
    obj_json = obj.to_json()
    assert isinstance(obj_json, str)


def test_from_json():
    obj = Wind.from_epw(EPW_OBJ)
    obj_json = obj.to_json()
    new_obj = Wind.from_json(obj_json)
    assert isinstance(new_obj, Wind)


def test_from_dataframe():
    obj = Wind.from_epw(EPW_OBJ)
    obj_df = obj.df
    new_obj = Wind.from_dataframe(obj_df, location=obj.location)
    assert isinstance(new_obj, Wind)


def test_from_average():
    obj = Wind.from_epw(EPW_OBJ)
    new_obj = Wind.from_average([obj, obj, obj])
    assert isinstance(new_obj, Wind)


# endregion: CLASS METHODS


# region: DUNDER METHODS
def test_len():
    obj = Wind.from_epw(EPW_OBJ)
    len(obj)


def test_str():
    obj = Wind.from_epw(EPW_OBJ)
    str(obj)


def test_repr():
    obj = Wind.from_epw(EPW_OBJ)
    repr(obj)


def test_hash():
    obj = Wind.from_epw(EPW_OBJ)
    hash(obj)


def test_eq():
    obj = Wind.from_epw(EPW_OBJ)
    obj == obj


def test_iter():
    obj = Wind.from_epw(EPW_OBJ)
    for i in obj:
        pass


def test_getitem():
    obj = Wind.from_epw(EPW_OBJ)
    obj[2]


def test_copy():
    obj = Wind.from_epw(EPW_OBJ)
    obj.__copy__()


# endregion: DUNDER METHODS


# region: PROPERTIES


def test_dates():
    obj = Wind.from_epw(EPW_OBJ)
    obj.dates


def test_start_date():
    obj = Wind.from_epw(EPW_OBJ)
    obj.start_date


def test_end_date():
    obj = Wind.from_epw(EPW_OBJ)
    obj.end_date


def test_lb_datetimes():
    obj = Wind.from_epw(EPW_OBJ)
    obj.lb_datetimes


def test_lb_dates():
    obj = Wind.from_epw(EPW_OBJ)
    obj.lb_dates


def test_datetimeindex():
    obj = Wind.from_epw(EPW_OBJ)
    obj.datetimeindex


def test_wind_speed_series():
    obj = Wind.from_epw(EPW_OBJ)
    obj.wind_speed_series


def test_wind_speed_collection():
    obj = Wind.from_epw(EPW_OBJ)
    obj.wind_speed_collection


def test_wind_direction_series():
    obj = Wind.from_epw(EPW_OBJ)
    obj.wind_direction_series


def test_wind_direction_collection():
    obj = Wind.from_epw(EPW_OBJ)
    obj.wind_direction_collection


def test_df():
    obj = Wind.from_epw(EPW_OBJ)
    obj.df


def test_analysis_period():
    obj = Wind.from_epw(EPW_OBJ)
    obj.analysis_period


def test_uv():
    obj = Wind.from_epw(EPW_OBJ)
    obj.uv


# endregion: PROPERTIES

# region: FILTERING


def test_filter_by_boolean_mask():
    obj = Wind.from_epw(EPW_OBJ)
    mask = (np.random.rand(8760) > 0.5).tolist()
    filtered_obj = obj.filter_by_boolean_mask(mask)
    assert isinstance(filtered_obj, Wind)
    assert len(filtered_obj) == sum(mask)


def test_filter_by_analysis_period():
    obj = Wind.from_epw(EPW_OBJ)
    ap = AnalysisPeriod(st_month=3, end_hour=18)
    filtered_obj = obj.filter_by_analysis_period(ap)
    assert isinstance(filtered_obj, Wind)
    assert len(filtered_obj) == len(ap)


def test_filter_by_time():
    obj = Wind.from_epw(EPW_OBJ)
    filtered_obj = obj.filter_by_time(hours=[2, 5, 6])
    assert isinstance(filtered_obj, Wind)


def test_filter_by_direction():
    # create wind object
    w = Wind.from_epw(EPW_OBJ)
    filtered_wind = w.filter_by_direction(left_angle=0, right_angle=180)
    assert isinstance(filtered_wind, Wind)


def test_filter_by_speed():
    w = Wind.from_epw(EPW_OBJ)
    filtered_wind = w.filter_by_speed(min_speed=2, max_speed=4)
    assert isinstance(filtered_wind, Wind)


# endregion: FILTERING


# region: INSTANCE
def test__direction_categories():
    obj = Wind.from_epw(EPW_OBJ)
    obj._direction_categories()


def test__direction_binned_data():
    obj = Wind.from_epw(EPW_OBJ)
    obj._direction_binned_data()


def test_proportion_calm():
    obj = Wind.from_epw(EPW_OBJ)
    obj.proportion_calm()


def test_calm_mask():
    obj = Wind.from_epw(EPW_OBJ)
    obj.calm_mask()


def test_percentile():
    obj = Wind.from_epw(EPW_OBJ)
    obj.percentile()


def test_to_height():
    obj = Wind.from_epw(EPW_OBJ)
    obj.to_height(target_height=15)


def test_apply_directional_factors():
    obj = Wind.from_epw(EPW_OBJ)
    obj.apply_directional_factors(directions=4, factors=[1, 2, 1, 2])


def test_direction_counts():
    obj = Wind.from_epw(EPW_OBJ)
    obj.direction_counts()


def test_prevailing():
    obj = Wind.from_epw(EPW_OBJ)
    obj.prevailing()


def test_month_hour_mean_matrix():
    obj = Wind.from_epw(EPW_OBJ)
    obj.month_hour_mean_matrix()


def test_windrose():
    obj = Wind.from_epw(EPW_OBJ)
    obj.windrose()


def test_histogram():
    obj = Wind.from_epw(EPW_OBJ)
    obj.histogram()


# endregion: INSTANCE


# region: VIZ
def test_plot_windprofile():
    obj = Wind.from_epw(EPW_OBJ)
    obj.plot_windprofile()


def test_plot_windmatrix():
    obj = Wind.from_epw(EPW_OBJ)
    obj.plot_windmatrix()


def test_plot_windrose():
    obj = Wind.from_epw(EPW_OBJ)
    obj.plot_windrose()


def test_plot_windhistogram():
    obj = Wind.from_epw(EPW_OBJ)
    obj.plot_windhistogram()


def test_plot_densityfunction():
    obj = Wind.from_epw(EPW_OBJ)
    obj.plot_densityfunction()


# endregion: VIZ

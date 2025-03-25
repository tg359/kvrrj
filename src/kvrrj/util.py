import warnings

import numpy as np


def wind_speed_at_height(
    reference_value: float,
    reference_height: float,
    target_height: float,
    **kwargs,
) -> float:
    """Calculate the wind speed at a given height from the 10m default height
        as stated in an EPW file.

    Args:
        reference_value (float):
            The speed to be translated.
        reference_height (float):
            The original height of the wind speed being translated.
        target_height (float):
            The target height of the wind speed being translated.

        **kwargs:
            Additional keyword arguments to pass to the translation method. These include:
            terrain_roughness_length (float):
                A value describing how rough the ground is. Default is
                0.03 for Open flat terrain; grass, few isolated obstacles.
            log_function (bool, optional):
                Set to True to used the log transformation method, or
                False for the exponent method. Defaults to True.

    Notes:
        Terrain roughness lengths can be found in the following table:

        +---------------------------------------------------+-----------+
        | Terrain description                               |  z0 (m)   |
        +===================================================+===========+
        | Open sea, Fetch at least 5 km                     |    0.0002 |
        +---------------------------------------------------+-----------+
        | Mud flats, snow; no vegetation, no obstacles      |    0.005  |
        +---------------------------------------------------+-----------+
        | Open flat terrain; grass, few isolated obstacle   |    0.03   |
        +---------------------------------------------------+-----------+
        | Low crops; occasional large obstacles, x/H > 20   |    0.10   |
        +---------------------------------------------------+-----------+
        | High crops; scattered obstacles, 15 < x/H < 20    |    0.25   |
        +---------------------------------------------------+-----------+
        | Parkland, bushes; numerous obstacles, x/H ≈ 10    |    0.5    |
        +---------------------------------------------------+-----------+
        | Regular large obstacle coverage (suburb, forest)  |    1.0    |
        +---------------------------------------------------+-----------+
        | City centre with high- and low-rise buildings     |    ≥ 2    |
        +---------------------------------------------------+-----------+


    Returns:
        float:
            The translated wind speed at the target height.
    """

    terrain_roughness_length = kwargs.get("terrain_roughness_length", 0.03)
    log_function = kwargs.get("log_function", True)
    kwargs = {}  # reset kwargs to remove invalid arguments

    if log_function:
        return reference_value * (
            np.log(target_height / terrain_roughness_length)
            / np.log(reference_height / terrain_roughness_length)
        )
    wind_shear_exponent = 1 / 7
    return reference_value * (
        np.power((target_height / reference_height), wind_shear_exponent)
    )


def temperature_at_height(
    reference_value: float,
    reference_height: float,
    target_height: float,
    **kwargs,
) -> float:
    """Estimate the dry-bulb temperature at a given height from a referenced
        dry-bulb temperature at another height.

    Args:
        reference_value (float):
            The temperature to translate.
        reference_height (float):
            The height of the reference temperature.
        target_height (float):
            The height to translate the reference temperature towards.
        **kwargs:
            Additional keyword arguments to pass to the translation method. These include:
            reduction_per_km_altitude_gain (float, optional):
                The lapse rate of the atmosphere. Defaults to 0.0065 based
                on https://scied.ucar.edu/learning-zone/atmosphere/change-atmosphere-altitude#:~:text=Near%20the%20Earth's%20surface%2C%20air,standard%20(average)%20lapse%20rate
            lapse_rate (float, optional):
                The degrees C reduction for every 1 altitude gain. Default is 0.0065C for clear
                conditions (or 6.5C per 1km). This would be nearer 0.0098C/m if cloudy/moist air conditions.

    Returns:
        float:
            A translated air temperature.
    """
    # pylint: enable=C0301

    if (target_height > 8000) or (reference_height > 8000):
        warning_str = (
            "The heights input into this calculation exist partially above "
            "the egde of the troposphere. This method is only valid below 8000m."
        )
        warnings.warn(warning_str)

    lapse_rate = kwargs.get("lapse_rate", 0.0065)
    kwargs = {}  # reset kwargs to remove invalid arguments

    height_difference = target_height - reference_height

    return reference_value - (height_difference * lapse_rate)


def radiation_at_height(
    reference_value: float,
    target_height: float,
    reference_height: float,
    **kwargs,
) -> float:
    """Calculate the radiation at a given height, given a reference
    radiation and height.

    References:
        Armel Oumbe, Lucien Wald. A parameterisation of vertical profile of
        solar irradiance for correcting solar fluxes for changes in terrain
        elevation. Earth Observation and Water Cycle Science Conference, Nov
        2009, Frascati, Italy. pp.S05.

    Args:
        reference_value (float):
            The radiation at the reference height.
        target_height (float):
            The height at which the radiation is required, in m.
        reference_height (float, optional):
            The height at which the reference radiation was measured.
        **kwargs:
            Additional keyword arguments to pass to the translation method. These include:
            lapse_rate (float, optional):
                The lapse rate of the atmosphere. Defaults to 0.08.

    Returns:
        float:
            The radiation at the given height.
    """
    lapse_rate = kwargs.get("lapse_rate", 0.08)
    kwargs = {}  # reset kwargs to remove invalid arguments

    lapse_rate_per_m = lapse_rate * reference_value / 1000
    increase = lapse_rate_per_m * (target_height - reference_height)
    return reference_value + increase


def air_pressure_at_height(
    reference_value: float,
    target_height: float,
    reference_height: float,
) -> float:
    """Calculate the air pressure at a given height, given a reference pressure and height.

    Args:
        reference_value (float):
            The pressure at the reference height, in Pa.
        target_height (float):
            The height at which the pressure is required, in m.
        reference_height (float, optional):
            The height at which the reference pressure was measured. Defaults to 10m.

    Returns:
        float:
            The pressure at the given height.
    """
    return (
        reference_value
        * (1 - 0.0065 * (target_height - reference_height) / 288.15) ** 5.255
    )

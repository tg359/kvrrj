# import all datatypes
import warnings

from ladybug.datatype import TYPESDICT
from ladybug.datatype.angle import Angle, WindDirection
from ladybug.datatype.area import Area
from ladybug.datatype.base import DataTypeBase
from ladybug.datatype.current import Current
from ladybug.datatype.distance import (
    CeilingHeight,
    Distance,
    LiquidPrecipitationDepth,
    PrecipitableWater,
    SnowDepth,
    Visibility,
)
from ladybug.datatype.energy import Energy
from ladybug.datatype.energyflux import (
    DiffuseHorizontalIrradiance,
    DirectHorizontalIrradiance,
    DirectNormalIrradiance,
    EffectiveRadiantField,
    EnergyFlux,
    GlobalHorizontalIrradiance,
    HorizontalInfraredRadiationIntensity,
    Irradiance,
    MetabolicRate,
)
from ladybug.datatype.energyintensity import (
    DiffuseHorizontalRadiation,
    DirectHorizontalRadiation,
    DirectNormalRadiation,
    EnergyIntensity,
    ExtraterrestrialDirectNormalRadiation,
    ExtraterrestrialHorizontalRadiation,
    GlobalHorizontalRadiation,
    Radiation,
)
from ladybug.datatype.fraction import (
    AerosolOpticalDepth,
    Albedo,
    Fraction,
    HumidityRatio,
    LiquidPrecipitationQuantity,
    OpaqueSkyCover,
    PercentagePeopleDissatisfied,
    RelativeHumidity,
    TotalSkyCover,
)
from ladybug.datatype.generic import GenericType
from ladybug.datatype.illuminance import (
    DiffuseHorizontalIlluminance,
    DirectNormalIlluminance,
    GlobalHorizontalIlluminance,
    Illuminance,
)
from ladybug.datatype.luminance import Luminance, ZenithLuminance
from ladybug.datatype.mass import Mass
from ladybug.datatype.massflowrate import MassFlowRate
from ladybug.datatype.power import ActivityLevel, Power
from ladybug.datatype.pressure import AtmosphericStationPressure, Pressure
from ladybug.datatype.rvalue import ClothingInsulation, RValue
from ladybug.datatype.specificenergy import Enthalpy, SpecificEnergy
from ladybug.datatype.speed import AirSpeed, Speed, WindSpeed
from ladybug.datatype.temperature import (
    AirTemperature,
    ClothingTemperature,
    CoreBodyTemperature,
    DewPointTemperature,
    DryBulbTemperature,
    GroundTemperature,
    HeatIndexTemperature,
    MeanRadiantTemperature,
    NeutralTemperature,
    OperativeTemperature,
    PhysiologicalEquivalentTemperature,
    PrevailingOutdoorTemperature,
    RadiantTemperature,
    SkinTemperature,
    SkyTemperature,
    StandardEffectiveTemperature,
    Temperature,
    UniversalThermalClimateIndex,
    WetBulbGlobeTemperature,
    WetBulbTemperature,
    WindChillTemperature,
)
from ladybug.datatype.temperaturedelta import (
    AirTemperatureDelta,
    OperativeTemperatureDelta,
    RadiantTemperatureDelta,
    TemperatureDelta,
)
from ladybug.datatype.temperaturetime import (
    CoolingDegreeTime,
    HeatingDegreeTime,
    TemperatureTime,
)
from ladybug.datatype.thermalcondition import (
    CoreTemperatureCategory,
    DiscomfortReason,
    PredictedMeanVote,
    ThermalComfort,
    ThermalCondition,
    ThermalConditionElevenPoint,
    ThermalConditionFivePoint,
    ThermalConditionNinePoint,
    ThermalConditionSevenPoint,
    UTCICategory,
)
from ladybug.datatype.time import Time
from ladybug.datatype.uvalue import ConvectionCoefficient, RadiantCoefficient, UValue
from ladybug.datatype.voltage import Voltage
from ladybug.datatype.volume import Volume
from ladybug.datatype.volumeflowrate import VolumeFlowRate
from ladybug.datatype.volumeflowrateintensity import VolumeFlowRateIntensity
from matplotlib.colors import Colormap, ListedColormap


def to_string(datatype: DataTypeBase, unit: str) -> str:
    """Convert a ladybug datatype to a string representation.

    Args:
        datatype (DataTypeBase):
            The ladybug datatype to convert.
        unit (str):
            The unit of the datatype.

    Returns:
        str:
            The string representation of the datatype.
    """

    return f"{datatype} ({unit})"


def to_datatype(text: str) -> DataTypeBase:
    """Convert a string to a ladybug datatype."""

    str_elements = text.split(" ")

    if (len(str_elements) < 2) or ("(" not in text) or (")" not in text):
        raise ValueError(
            "The string to be converted into a LB Datatype must be in the format 'variable (unit)'"
        )

    str_elements = text.split(" ")
    unit = str_elements[-1].replace("(", "").replace(")", "")
    data_type = " ".join(str_elements[:-1])

    try:
        return TYPESDICT[data_type.replace(" ", "")]()
    except KeyError:
        warnings.warn(
            f"Datatype {data_type} not found in ladybug library. Returning generic datatype."
        )
        return GenericType(name=data_type, unit=unit)


def to_color(datatype: DataTypeBase) -> str:
    """Convert a ladybug datatype to a color.

    Args:
        datatype (DataTypeBase):
            The ladybug datatype to convert.

    Returns:
        str:
            The color representation of the datatype in hex format, including alpha.
    """

    # TODO - SET CORRECT COLORS
    color_lookup = {
        Angle: "#000000FF",  # black
        WindDirection: "#000000FF",  # black
        Area: "#000000FF",  # black
        Current: "#000000FF",  # black
        CeilingHeight: "#000000FF",  # black
        Distance: "#000000FF",  # black
        LiquidPrecipitationDepth: "#000000FF",  # black
        PrecipitableWater: "#000000FF",  # black
        SnowDepth: "#000000FF",  # black
        Visibility: "#000000FF",  # black
        Energy: "#000000FF",  # black
        DiffuseHorizontalIrradiance: "#000000FF",  # black
        DirectHorizontalIrradiance: "#000000FF",  # black
        DirectNormalIrradiance: "#000000FF",  # black
        EffectiveRadiantField: "#000000FF",  # black
        EnergyFlux: "#000000FF",  # black
        GlobalHorizontalIrradiance: "#000000FF",  # black
        HorizontalInfraredRadiationIntensity: "#000000FF",  # black
        Irradiance: "#000000FF",  # black
        MetabolicRate: "#000000FF",  # black
        DiffuseHorizontalRadiation: "#000000FF",  # black
        DirectHorizontalRadiation: "#000000FF",  # black
        DirectNormalRadiation: "#000000FF",  # black
        EnergyIntensity: "#000000FF",  # black
        ExtraterrestrialDirectNormalRadiation: "#000000FF",  # black
        ExtraterrestrialHorizontalRadiation: "#000000FF",  # black
        GlobalHorizontalRadiation: "#000000FF",  # black
        Radiation: "#000000FF",  # black
        AerosolOpticalDepth: "#000000FF",  # black
        Albedo: "#000000FF",  # black
        Fraction: "#000000FF",  # black
        HumidityRatio: "#000000FF",  # black
        LiquidPrecipitationQuantity: "#000000FF",  # black
        OpaqueSkyCover: "#000000FF",  # black
        PercentagePeopleDissatisfied: "#000000FF",  # black
        RelativeHumidity: "#000000FF",  # black
        TotalSkyCover: "#000000FF",  # black
        DiffuseHorizontalIlluminance: "#000000FF",  # black
        DirectNormalIlluminance: "#000000FF",  # black
        GlobalHorizontalIlluminance: "#000000FF",  # black
        Illuminance: "#000000FF",  # black
        Luminance: "#000000FF",  # black
        ZenithLuminance: "#000000FF",  # black
        Mass: "#000000FF",  # black
        MassFlowRate: "#000000FF",  # black
        ActivityLevel: "#000000FF",  # black
        Power: "#000000FF",  # black
        AtmosphericStationPressure: "#000000FF",  # black
        Pressure: "#000000FF",  # black
        ClothingInsulation: "#000000FF",  # black
        RValue: "#000000FF",  # black
        Enthalpy: "#000000FF",  # black
        SpecificEnergy: "#000000FF",  # black
        AirSpeed: "#000000FF",  # black
        Speed: "#000000FF",  # black
        WindSpeed: "#000000FF",  # black
        AirTemperature: "#000000FF",  # black
        ClothingTemperature: "#000000FF",  # black
        CoreBodyTemperature: "#000000FF",  # black
        DewPointTemperature: "#000000FF",  # black
        DryBulbTemperature: "#000000FF",  # black
        GroundTemperature: "#000000FF",  # black
        HeatIndexTemperature: "#000000FF",  # black
        MeanRadiantTemperature: "#000000FF",  # black
        NeutralTemperature: "#000000FF",  # black
        OperativeTemperature: "#000000FF",  # black
        PhysiologicalEquivalentTemperature: "#000000FF",  # black
        PrevailingOutdoorTemperature: "#000000FF",  # black
        RadiantTemperature: "#000000FF",  # black
        SkinTemperature: "#000000FF",  # black
        SkyTemperature: "#000000FF",  # black
        StandardEffectiveTemperature: "#000000FF",  # black
        Temperature: "#000000FF",  # black
        UniversalThermalClimateIndex: "#000000FF",  # black
        WetBulbGlobeTemperature: "#000000FF",  # black
        WetBulbTemperature: "#000000FF",  # black
        WindChillTemperature: "#000000FF",  # black
        AirTemperatureDelta: "#000000FF",  # black
        OperativeTemperatureDelta: "#000000FF",  # black
        RadiantTemperatureDelta: "#000000FF",  # black
        TemperatureDelta: "#000000FF",  # black
        CoolingDegreeTime: "#000000FF",  # black
        HeatingDegreeTime: "#000000FF",  # black
        TemperatureTime: "#000000FF",  # black
        CoreTemperatureCategory: "#000000FF",  # black
        DiscomfortReason: "#000000FF",  # black
        PredictedMeanVote: "#000000FF",  # black
        ThermalComfort: "#000000FF",  # black
        ThermalCondition: "#000000FF",  # black
        ThermalConditionElevenPoint: "#000000FF",  # black
        ThermalConditionFivePoint: "#000000FF",  # black
        ThermalConditionNinePoint: "#000000FF",  # black
        ThermalConditionSevenPoint: "#000000FF",  # black
        UTCICategory: "#000000FF",  # black
        Time: "#000000FF",  # black
        ConvectionCoefficient: "#000000FF",  # black
        RadiantCoefficient: "#000000FF",  # black
        UValue: "#000000FF",  # black
        Voltage: "#000000FF",  # black
        Volume: "#000000FF",  # black
        VolumeFlowRate: "#000000FF",  # black
        VolumeFlowRateIntensity: "#000000FF",  # black
    }

    try:
        return color_lookup[datatype]
    except KeyError:
        raise ValueError(f"Colormap not defined for datatype {datatype}.")


def to_colormap(datatype: DataTypeBase) -> Colormap:
    """Convert a ladybug datatype to a colormap."""

    # TODO - SET CORRECT COLORS
    colormap_lookup = {
        Angle: ListedColormap(colors=["black", "white"], name="Angle"),
        WindDirection: ListedColormap(colors=["black", "white"], name="WindDirection"),
        Area: ListedColormap(colors=["black", "white"], name="Area"),
        Current: ListedColormap(colors=["black", "white"], name="Current"),
        CeilingHeight: ListedColormap(colors=["black", "white"], name="CeilingHeight"),
        Distance: ListedColormap(colors=["black", "white"], name="Distance"),
        LiquidPrecipitationDepth: ListedColormap(
            colors=["black", "white"], name="LiquidPrecipitationDepth"
        ),
        PrecipitableWater: ListedColormap(
            colors=["black", "white"], name="PrecipitableWater"
        ),
        SnowDepth: ListedColormap(colors=["black", "white"], name="SnowDepth"),
        Visibility: ListedColormap(colors=["black", "white"], name="Visibility"),
        Energy: ListedColormap(colors=["black", "white"], name="Energy"),
        DiffuseHorizontalIrradiance: ListedColormap(
            colors=["black", "white"], name="DiffuseHorizontalIrradiance"
        ),
        DirectHorizontalIrradiance: ListedColormap(
            colors=["black", "white"], name="DirectHorizontalIrradiance"
        ),
        DirectNormalIrradiance: ListedColormap(
            colors=["black", "white"], name="DirectNormalIrradiance"
        ),
        EffectiveRadiantField: ListedColormap(
            colors=["black", "white"], name="EffectiveRadiantField"
        ),
        EnergyFlux: ListedColormap(colors=["black", "white"], name="EnergyFlux"),
        GlobalHorizontalIrradiance: ListedColormap(
            colors=["black", "white"], name="GlobalHorizontalIrradiance"
        ),
        HorizontalInfraredRadiationIntensity: ListedColormap(
            colors=["black", "white"], name="HorizontalInfraredRadiationIntensity"
        ),
        Irradiance: ListedColormap(colors=["black", "white"], name="Irradiance"),
        MetabolicRate: ListedColormap(colors=["black", "white"], name="MetabolicRate"),
        DiffuseHorizontalRadiation: ListedColormap(
            colors=["black", "white"], name="DiffuseHorizontalRadiation"
        ),
        DirectHorizontalRadiation: ListedColormap(
            colors=["black", "white"], name="DirectHorizontalRadiation"
        ),
        DirectNormalRadiation: ListedColormap(
            colors=["black", "white"], name="DirectNormalRadiation"
        ),
        EnergyIntensity: ListedColormap(
            colors=["black", "white"], name="EnergyIntensity"
        ),
        ExtraterrestrialDirectNormalRadiation: ListedColormap(
            colors=["black", "white"], name="ExtraterrestrialDirectNormalRadiation"
        ),
        ExtraterrestrialHorizontalRadiation: ListedColormap(
            colors=["black", "white"], name="ExtraterrestrialHorizontalRadiation"
        ),
        GlobalHorizontalRadiation: ListedColormap(
            colors=["black", "white"], name="GlobalHorizontalRadiation"
        ),
        Radiation: ListedColormap(colors=["black", "white"], name="Radiation"),
        AerosolOpticalDepth: ListedColormap(
            colors=["black", "white"], name="AerosolOpticalDepth"
        ),
        Albedo: ListedColormap(colors=["black", "white"], name="Albedo"),
        Fraction: ListedColormap(colors=["black", "white"], name="Fraction"),
        HumidityRatio: ListedColormap(colors=["black", "white"], name="HumidityRatio"),
        LiquidPrecipitationQuantity: ListedColormap(
            colors=["black", "white"], name="LiquidPrecipitationQuantity"
        ),
        OpaqueSkyCover: ListedColormap(
            colors=["black", "white"], name="OpaqueSkyCover"
        ),
        PercentagePeopleDissatisfied: ListedColormap(
            colors=["black", "white"], name="PercentagePeopleDissatisfied"
        ),
        RelativeHumidity: ListedColormap(
            colors=["black", "white"], name="RelativeHumidity"
        ),
        TotalSkyCover: ListedColormap(colors=["black", "white"], name="TotalSkyCover"),
        DiffuseHorizontalIlluminance: ListedColormap(
            colors=["black", "white"], name="DiffuseHorizontalIlluminance"
        ),
        DirectNormalIlluminance: ListedColormap(
            colors=["black", "white"], name="DirectNormalIlluminance"
        ),
        GlobalHorizontalIlluminance: ListedColormap(
            colors=["black", "white"], name="GlobalHorizontalIlluminance"
        ),
        Illuminance: ListedColormap(colors=["black", "white"], name="Illuminance"),
        Luminance: ListedColormap(colors=["black", "white"], name="Luminance"),
        ZenithLuminance: ListedColormap(
            colors=["black", "white"], name="ZenithLuminance"
        ),
        Mass: ListedColormap(colors=["black", "white"], name="Mass"),
        MassFlowRate: ListedColormap(colors=["black", "white"], name="MassFlowRate"),
        ActivityLevel: ListedColormap(colors=["black", "white"], name="ActivityLevel"),
        Power: ListedColormap(colors=["black", "white"], name="Power"),
        AtmosphericStationPressure: ListedColormap(
            colors=["black", "white"], name="AtmosphericStationPressure"
        ),
        Pressure: ListedColormap(colors=["black", "white"], name="Pressure"),
        ClothingInsulation: ListedColormap(
            colors=["black", "white"], name="ClothingInsulation"
        ),
        RValue: ListedColormap(colors=["black", "white"], name="RValue"),
        Enthalpy: ListedColormap(colors=["black", "white"], name="Enthalpy"),
        SpecificEnergy: ListedColormap(
            colors=["black", "white"], name="SpecificEnergy"
        ),
        AirSpeed: ListedColormap(colors=["black", "white"], name="AirSpeed"),
        Speed: ListedColormap(colors=["black", "white"], name="Speed"),
        WindSpeed: ListedColormap(colors=["black", "white"], name="WindSpeed"),
        AirTemperature: ListedColormap(
            colors=["black", "white"], name="AirTemperature"
        ),
        ClothingTemperature: ListedColormap(
            colors=["black", "white"], name="ClothingTemperature"
        ),
        CoreBodyTemperature: ListedColormap(
            colors=["black", "white"], name="CoreBodyTemperature"
        ),
        DewPointTemperature: ListedColormap(
            colors=["black", "white"], name="DewPointTemperature"
        ),
        DryBulbTemperature: ListedColormap(
            colors=["black", "white"], name="DryBulbTemperature"
        ),
        GroundTemperature: ListedColormap(
            colors=["black", "white"], name="GroundTemperature"
        ),
        HeatIndexTemperature: ListedColormap(
            colors=["black", "white"], name="HeatIndexTemperature"
        ),
        MeanRadiantTemperature: ListedColormap(
            colors=["black", "white"], name="MeanRadiantTemperature"
        ),
        NeutralTemperature: ListedColormap(
            colors=["black", "white"], name="NeutralTemperature"
        ),
        OperativeTemperature: ListedColormap(
            colors=["black", "white"], name="OperativeTemperature"
        ),
        PhysiologicalEquivalentTemperature: ListedColormap(
            colors=["black", "white"], name="PhysiologicalEquivalentTemperature"
        ),
        PrevailingOutdoorTemperature: ListedColormap(
            colors=["black", "white"], name="PrevailingOutdoorTemperature"
        ),
        RadiantTemperature: ListedColormap(
            colors=["black", "white"], name="RadiantTemperature"
        ),
        SkinTemperature: ListedColormap(
            colors=["black", "white"], name="SkinTemperature"
        ),
        SkyTemperature: ListedColormap(
            colors=["black", "white"], name="SkyTemperature"
        ),
        StandardEffectiveTemperature: ListedColormap(
            colors=["black", "white"], name="StandardEffectiveTemperature"
        ),
        Temperature: ListedColormap(colors=["black", "white"], name="Temperature"),
        UniversalThermalClimateIndex: ListedColormap(
            colors=["black", "white"], name="UniversalThermalClimateIndex"
        ),
        WetBulbGlobeTemperature: ListedColormap(
            colors=["black", "white"], name="WetBulbGlobeTemperature"
        ),
        WetBulbTemperature: ListedColormap(
            colors=["black", "white"], name="WetBulbTemperature"
        ),
        WindChillTemperature: ListedColormap(
            colors=["black", "white"], name="WindChillTemperature"
        ),
        AirTemperatureDelta: ListedColormap(
            colors=["black", "white"], name="AirTemperatureDelta"
        ),
        OperativeTemperatureDelta: ListedColormap(
            colors=["black", "white"], name="OperativeTemperatureDelta"
        ),
        RadiantTemperatureDelta: ListedColormap(
            colors=["black", "white"], name="RadiantTemperatureDelta"
        ),
        TemperatureDelta: ListedColormap(
            colors=["black", "white"], name="TemperatureDelta"
        ),
        CoolingDegreeTime: ListedColormap(
            colors=["black", "white"], name="CoolingDegreeTime"
        ),
        HeatingDegreeTime: ListedColormap(
            colors=["black", "white"], name="HeatingDegreeTime"
        ),
        TemperatureTime: ListedColormap(
            colors=["black", "white"], name="TemperatureTime"
        ),
        CoreTemperatureCategory: ListedColormap(
            colors=["black", "white"], name="CoreTemperatureCategory"
        ),
        DiscomfortReason: ListedColormap(
            colors=["black", "white"], name="DiscomfortReason"
        ),
        PredictedMeanVote: ListedColormap(
            colors=["black", "white"], name="PredictedMeanVote"
        ),
        ThermalComfort: ListedColormap(
            colors=["black", "white"], name="ThermalComfort"
        ),
        ThermalCondition: ListedColormap(
            colors=["black", "white"], name="ThermalCondition"
        ),
        ThermalConditionElevenPoint: ListedColormap(
            colors=["black", "white"], name="ThermalConditionElevenPoint"
        ),
        ThermalConditionFivePoint: ListedColormap(
            colors=["black", "white"], name="ThermalConditionFivePoint"
        ),
        ThermalConditionNinePoint: ListedColormap(
            colors=["black", "white"], name="ThermalConditionNinePoint"
        ),
        ThermalConditionSevenPoint: ListedColormap(
            colors=["black", "white"], name="ThermalConditionSevenPoint"
        ),
        UTCICategory: ListedColormap(colors=["black", "white"], name="UTCICategory"),
        Time: ListedColormap(colors=["black", "white"], name="Time"),
        ConvectionCoefficient: ListedColormap(
            colors=["black", "white"], name="ConvectionCoefficient"
        ),
        RadiantCoefficient: ListedColormap(
            colors=["black", "white"], name="RadiantCoefficient"
        ),
        UValue: ListedColormap(colors=["black", "white"], name="UValue"),
        Voltage: ListedColormap(colors=["black", "white"], name="Voltage"),
        Volume: ListedColormap(colors=["black", "white"], name="Volume"),
        VolumeFlowRate: ListedColormap(
            colors=["black", "white"], name="VolumeFlowRate"
        ),
        VolumeFlowRateIntensity: ListedColormap(
            colors=["black", "white"], name="VolumeFlowRateIntensity"
        ),
    }

    try:
        return colormap_lookup[datatype]
    except KeyError:
        raise ValueError(f"Colormap not defined for datatype {datatype}.")


# TODO -= update wind and solar methods!

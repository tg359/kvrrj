"""Default colors and colormap for ladybug datatypes."""

import colorsys
import re
from typing import Any

import numpy as np
from ladybug.color import Color
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
from matplotlib.colors import (
    Colormap,
    LinearSegmentedColormap,
    ListedColormap,
    colorConverter,
    to_rgba,
)

PARULA_COLORMAP = LinearSegmentedColormap.from_list(
    name="parula",
    colors=[
        [0.2422, 0.1504, 0.6603],
        [0.2444, 0.1534, 0.6728],
        [0.2464, 0.1569, 0.6847],
        [0.2484, 0.1607, 0.6961],
        [0.2503, 0.1648, 0.7071],
        [0.2522, 0.1689, 0.7179],
        [0.254, 0.1732, 0.7286],
        [0.2558, 0.1773, 0.7393],
        [0.2576, 0.1814, 0.7501],
        [0.2594, 0.1854, 0.761],
        [0.2611, 0.1893, 0.7719],
        [0.2628, 0.1932, 0.7828],
        [0.2645, 0.1972, 0.7937],
        [0.2661, 0.2011, 0.8043],
        [0.2676, 0.2052, 0.8148],
        [0.2691, 0.2094, 0.8249],
        [0.2704, 0.2138, 0.8346],
        [0.2717, 0.2184, 0.8439],
        [0.2729, 0.2231, 0.8528],
        [0.274, 0.228, 0.8612],
        [0.2749, 0.233, 0.8692],
        [0.2758, 0.2382, 0.8767],
        [0.2766, 0.2435, 0.884],
        [0.2774, 0.2489, 0.8908],
        [0.2781, 0.2543, 0.8973],
        [0.2788, 0.2598, 0.9035],
        [0.2794, 0.2653, 0.9094],
        [0.2798, 0.2708, 0.915],
        [0.2802, 0.2764, 0.9204],
        [0.2806, 0.2819, 0.9255],
        [0.2809, 0.2875, 0.9305],
        [0.2811, 0.293, 0.9352],
        [0.2813, 0.2985, 0.9397],
        [0.2814, 0.304, 0.9441],
        [0.2814, 0.3095, 0.9483],
        [0.2813, 0.315, 0.9524],
        [0.2811, 0.3204, 0.9563],
        [0.2809, 0.3259, 0.96],
        [0.2807, 0.3313, 0.9636],
        [0.2803, 0.3367, 0.967],
        [0.2798, 0.3421, 0.9702],
        [0.2791, 0.3475, 0.9733],
        [0.2784, 0.3529, 0.9763],
        [0.2776, 0.3583, 0.9791],
        [0.2766, 0.3638, 0.9817],
        [0.2754, 0.3693, 0.984],
        [0.2741, 0.3748, 0.9862],
        [0.2726, 0.3804, 0.9881],
        [0.271, 0.386, 0.9898],
        [0.2691, 0.3916, 0.9912],
        [0.267, 0.3973, 0.9924],
        [0.2647, 0.403, 0.9935],
        [0.2621, 0.4088, 0.9946],
        [0.2591, 0.4145, 0.9955],
        [0.2556, 0.4203, 0.9965],
        [0.2517, 0.4261, 0.9974],
        [0.2473, 0.4319, 0.9983],
        [0.2424, 0.4378, 0.9991],
        [0.2369, 0.4437, 0.9996],
        [0.2311, 0.4497, 0.9995],
        [0.225, 0.4559, 0.9985],
        [0.2189, 0.462, 0.9968],
        [0.2128, 0.4682, 0.9948],
        [0.2066, 0.4743, 0.9926],
        [0.2006, 0.4803, 0.9906],
        [0.195, 0.4861, 0.9887],
        [0.1903, 0.4919, 0.9867],
        [0.1869, 0.4975, 0.9844],
        [0.1847, 0.503, 0.9819],
        [0.1831, 0.5084, 0.9793],
        [0.1818, 0.5138, 0.9766],
        [0.1806, 0.5191, 0.9738],
        [0.1795, 0.5244, 0.9709],
        [0.1785, 0.5296, 0.9677],
        [0.1778, 0.5349, 0.9641],
        [0.1773, 0.5401, 0.9602],
        [0.1768, 0.5452, 0.956],
        [0.1764, 0.5504, 0.9516],
        [0.1755, 0.5554, 0.9473],
        [0.174, 0.5605, 0.9432],
        [0.1716, 0.5655, 0.9393],
        [0.1686, 0.5705, 0.9357],
        [0.1649, 0.5755, 0.9323],
        [0.161, 0.5805, 0.9289],
        [0.1573, 0.5854, 0.9254],
        [0.154, 0.5902, 0.9218],
        [0.1513, 0.595, 0.9182],
        [0.1492, 0.5997, 0.9147],
        [0.1475, 0.6043, 0.9113],
        [0.1461, 0.6089, 0.908],
        [0.1446, 0.6135, 0.905],
        [0.1429, 0.618, 0.9022],
        [0.1408, 0.6226, 0.8998],
        [0.1383, 0.6272, 0.8975],
        [0.1354, 0.6317, 0.8953],
        [0.1321, 0.6363, 0.8932],
        [0.1288, 0.6408, 0.891],
        [0.1253, 0.6453, 0.8887],
        [0.1219, 0.6497, 0.8862],
        [0.1185, 0.6541, 0.8834],
        [0.1152, 0.6584, 0.8804],
        [0.1119, 0.6627, 0.877],
        [0.1085, 0.6669, 0.8734],
        [0.1048, 0.671, 0.8695],
        [0.1009, 0.675, 0.8653],
        [0.0964, 0.6789, 0.8609],
        [0.0914, 0.6828, 0.8562],
        [0.0855, 0.6865, 0.8513],
        [0.0789, 0.6902, 0.8462],
        [0.0713, 0.6938, 0.8409],
        [0.0628, 0.6972, 0.8355],
        [0.0535, 0.7006, 0.8299],
        [0.0433, 0.7039, 0.8242],
        [0.0328, 0.7071, 0.8183],
        [0.0234, 0.7103, 0.8124],
        [0.0155, 0.7133, 0.8064],
        [0.0091, 0.7163, 0.8003],
        [0.0046, 0.7192, 0.7941],
        [0.0019, 0.722, 0.7878],
        [0.0009, 0.7248, 0.7815],
        [0.0018, 0.7275, 0.7752],
        [0.0046, 0.7301, 0.7688],
        [0.0094, 0.7327, 0.7623],
        [0.0162, 0.7352, 0.7558],
        [0.0253, 0.7376, 0.7492],
        [0.0369, 0.74, 0.7426],
        [0.0504, 0.7423, 0.7359],
        [0.0638, 0.7446, 0.7292],
        [0.077, 0.7468, 0.7224],
        [0.0899, 0.7489, 0.7156],
        [0.1023, 0.751, 0.7088],
        [0.1141, 0.7531, 0.7019],
        [0.1252, 0.7552, 0.695],
        [0.1354, 0.7572, 0.6881],
        [0.1448, 0.7593, 0.6812],
        [0.1532, 0.7614, 0.6741],
        [0.1609, 0.7635, 0.6671],
        [0.1678, 0.7656, 0.6599],
        [0.1741, 0.7678, 0.6527],
        [0.1799, 0.7699, 0.6454],
        [0.1853, 0.7721, 0.6379],
        [0.1905, 0.7743, 0.6303],
        [0.1954, 0.7765, 0.6225],
        [0.2003, 0.7787, 0.6146],
        [0.2061, 0.7808, 0.6065],
        [0.2118, 0.7828, 0.5983],
        [0.2178, 0.7849, 0.5899],
        [0.2244, 0.7869, 0.5813],
        [0.2318, 0.7887, 0.5725],
        [0.2401, 0.7905, 0.5636],
        [0.2491, 0.7922, 0.5546],
        [0.2589, 0.7937, 0.5454],
        [0.2695, 0.7951, 0.536],
        [0.2809, 0.7964, 0.5266],
        [0.2929, 0.7975, 0.517],
        [0.3052, 0.7985, 0.5074],
        [0.3176, 0.7994, 0.4975],
        [0.3301, 0.8002, 0.4876],
        [0.3424, 0.8009, 0.4774],
        [0.3548, 0.8016, 0.4669],
        [0.3671, 0.8021, 0.4563],
        [0.3795, 0.8026, 0.4454],
        [0.3921, 0.8029, 0.4344],
        [0.405, 0.8031, 0.4233],
        [0.4184, 0.803, 0.4122],
        [0.4322, 0.8028, 0.4013],
        [0.4463, 0.8024, 0.3904],
        [0.4608, 0.8018, 0.3797],
        [0.4753, 0.8011, 0.3691],
        [0.4899, 0.8002, 0.3586],
        [0.5044, 0.7993, 0.348],
        [0.5187, 0.7982, 0.3374],
        [0.5329, 0.797, 0.3267],
        [0.547, 0.7957, 0.3159],
        [0.5609, 0.7943, 0.305],
        [0.5748, 0.7929, 0.2941],
        [0.5886, 0.7913, 0.2833],
        [0.6024, 0.7896, 0.2726],
        [0.6161, 0.7878, 0.2622],
        [0.6297, 0.7859, 0.2521],
        [0.6433, 0.7839, 0.2423],
        [0.6567, 0.7818, 0.2329],
        [0.6701, 0.7796, 0.2239],
        [0.6833, 0.7773, 0.2155],
        [0.6963, 0.775, 0.2075],
        [0.7091, 0.7727, 0.1998],
        [0.7218, 0.7703, 0.1924],
        [0.7344, 0.7679, 0.1852],
        [0.7468, 0.7654, 0.1782],
        [0.759, 0.7629, 0.1717],
        [0.771, 0.7604, 0.1658],
        [0.7829, 0.7579, 0.1608],
        [0.7945, 0.7554, 0.157],
        [0.806, 0.7529, 0.1546],
        [0.8172, 0.7505, 0.1535],
        [0.8281, 0.7481, 0.1536],
        [0.8389, 0.7457, 0.1546],
        [0.8495, 0.7435, 0.1564],
        [0.86, 0.7413, 0.1587],
        [0.8703, 0.7392, 0.1615],
        [0.8804, 0.7372, 0.165],
        [0.8903, 0.7353, 0.1695],
        [0.9, 0.7336, 0.1749],
        [0.9093, 0.7321, 0.1815],
        [0.9184, 0.7308, 0.189],
        [0.9272, 0.7298, 0.1973],
        [0.9357, 0.729, 0.2061],
        [0.944, 0.7285, 0.2151],
        [0.9523, 0.7284, 0.2237],
        [0.9606, 0.7285, 0.2312],
        [0.9689, 0.7292, 0.2373],
        [0.977, 0.7304, 0.2418],
        [0.9842, 0.733, 0.2446],
        [0.99, 0.7365, 0.2429],
        [0.9946, 0.7407, 0.2394],
        [0.9966, 0.7458, 0.2351],
        [0.9971, 0.7513, 0.2309],
        [0.9972, 0.7569, 0.2267],
        [0.9971, 0.7626, 0.2224],
        [0.9969, 0.7683, 0.2181],
        [0.9966, 0.774, 0.2138],
        [0.9962, 0.7798, 0.2095],
        [0.9957, 0.7856, 0.2053],
        [0.9949, 0.7915, 0.2012],
        [0.9938, 0.7974, 0.1974],
        [0.9923, 0.8034, 0.1939],
        [0.9906, 0.8095, 0.1906],
        [0.9885, 0.8156, 0.1875],
        [0.9861, 0.8218, 0.1846],
        [0.9835, 0.828, 0.1817],
        [0.9807, 0.8342, 0.1787],
        [0.9778, 0.8404, 0.1757],
        [0.9748, 0.8467, 0.1726],
        [0.972, 0.8529, 0.1695],
        [0.9694, 0.8591, 0.1665],
        [0.9671, 0.8654, 0.1636],
        [0.9651, 0.8716, 0.1608],
        [0.9634, 0.8778, 0.1582],
        [0.9619, 0.884, 0.1557],
        [0.9608, 0.8902, 0.1532],
        [0.9601, 0.8963, 0.1507],
        [0.9596, 0.9023, 0.148],
        [0.9595, 0.9084, 0.145],
        [0.9597, 0.9143, 0.1418],
        [0.9601, 0.9203, 0.1382],
        [0.9608, 0.9262, 0.1344],
        [0.9618, 0.932, 0.1304],
        [0.9629, 0.9379, 0.1261],
        [0.9642, 0.9437, 0.1216],
        [0.9657, 0.9494, 0.1168],
        [0.9674, 0.9552, 0.1116],
        [0.9692, 0.9609, 0.1061],
        [0.9711, 0.9667, 0.1001],
        [0.973, 0.9724, 0.0938],
        [0.9749, 0.9782, 0.0872],
        [0.9769, 0.9839, 0.0805],
    ],
)


def _datatype_to_color(datatype: DataTypeBase) -> str:
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
        GenericType: "#000000FF",  # black
    }

    try:
        return color_lookup[datatype]
    except KeyError:
        raise ValueError(f"Colormap not defined for datatype {datatype}.")


def _datatype_to_colormap(datatype: DataTypeBase) -> Colormap:
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
        GenericType: ListedColormap(colors=["black", "white"], name="GenericType"),
    }

    try:
        return colormap_lookup[datatype]
    except KeyError:
        raise ValueError(f"Colormap not defined for datatype {datatype}.")


def _lb_color_to_rgba(color: Color) -> tuple[float, float, float, float]:
    """Convert a ladybug color to an RGBA tuple.

    Args:
        color (Color):
            The ladybug color to convert.

    Returns:
        tuple[float]:
            The RGBA tuple.
    """
    return (
        float(color.r / 255),
        float(color.g / 255),
        float(color.b / 255),
        float(color.a / 255),
    )


def _plotly_color_to_rgba(color: str) -> tuple[float, float, float, float]:
    """Convert a plotly color to an RGBA tuple.

    Args:
        color (str):
            The plotly color to convert.

    Returns:
        tuple[float]:
            The RGBA tuple.
    """
    pattern = r"^rgba?\((\d{1,3}),\s*(\d{1,3}),\s*(\d{1,3})(?:,\s*([\d\.]+))?\)$"
    match = re.match(pattern, color)
    if not match:
        raise ValueError("String does not match the required format.")
    r, g, b, a = match.groups()
    a = float(a) if a is not None else 1.0
    return to_rgba([r / 255, g / 255, b / 255, a])


def _mpl_colormap_to_colorset(cmap: Colormap, N: int = 11) -> tuple[Color]:
    """Convert a matplotlib colormap to a ladybug colorset.

    Args:
        cmap (Colormap):
            The matplotlib colormap to convert.
        N (int, optional):
            The number of colors to sample from the colormap. Defaults to 11.

    Returns:
        Colorset:
            The ladybug colorset.
    """
    return [tuple(cmap(i)) for i in np.linspace(0, 1, N)]


def relative_luminance(color: Any):
    """Calculate the relative luminance of a color according to W3C standards

    Args:
        color (Any):
            matplotlib color or sequence of matplotlib colors - Hex code,
            rgb-tuple, or html color name.

    Returns:
        float:
            Luminance value between 0 and 1.
    """
    rgb = colorConverter.to_rgba_array(color)[:, :3]
    rgb = np.where(rgb <= 0.03928, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    lum = rgb.dot([0.2126, 0.7152, 0.0722])
    try:
        return lum.item()
    except ValueError:
        return lum


def contrasting_color(color: Any):
    """Calculate the contrasting color for a given color.

    Args:
        color (Any):
            matplotlib color or sequence of matplotlib colors - Hex code,
            rgb-tuple, or html color name.

    Returns:
        str:
            String code of the contrasting color.
    """
    return ".15" if relative_luminance(color) > 0.408 else "w"


def lighten_color(
    color: str | tuple, amount: float = 0.5
) -> tuple[float, float, float]:
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.

    Args:
        color (str):
            A color-like string.
        amount (float):
            The amount of lightening to apply.

    Returns:
        tuple[float]:
            An RGB value.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """

    c = colorsys.rgb_to_hls(*to_rgba(color)[:-1])
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def average_color(
    colors: Any,
    keep_alpha: bool = False,
    weights: list[float] = None,  # type: ignore
) -> tuple[float, float, float, float]:
    """Return the average color from a list of colors.

    Args:
        colors (Any):
            A list of colors.
        keep_alpha (bool, optional):
            If True, the alpha value of the color is kept. Defaults to False.
        weights (list[float], optional):
            A list of weights for each color. Defaults to None.

    Returns:
        color: str
            The average color in hex format.
    """

    if not isinstance(colors, (list, tuple)):
        raise ValueError("colors must be a list")

    if len(colors) == 1:
        return colors[0]

    return to_rgba(
        np.average([to_rgba(c) for c in colors], axis=0, weights=weights),
    )

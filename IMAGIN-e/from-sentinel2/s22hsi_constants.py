from enum import Enum

import numpy as np

BBOX_SIZE = 46080
HSI_PIX_SIZE = 45

HSI_BANDS = [f"BAND_{cw}" for cw in np.arange(450, 950, 10)]
HSI_CW = np.arange(450, 950, 10)
HSI_IRR = np.array(
    [
        2119.93,
        2084.97,
        2018.61,
        2123.95,
        2050.55,
        1954.17,
        1926.65,
        1838.33,
        1919.48,
        1833.65,
        1904.1,
        1794.67,
        1805.18,
        1821.8,
        1714.66,
        1765.75,
        1713.2,
        1697.86,
        1667.03,
        1622.64,
        1591.74,
        1560.34,
        1537.84,
        1497.69,
        1494.47,
        1462.7,
        1420.19,
        1371.8,
        1340.24,
        1290.1,
        1283.42,
        1280.09,
        1242.99,
        1178.18,
        1171.28,
        1132.17,
        1112.9,
        1077.67,
        1083.16,
        1081.15,
        978.98,
        1031.68,
        1010.42,
        948.43,
        935.72,
        889.41,
        871.92,
        863.06,
        844.94,
        825.71,
    ]
)

S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09"]
S2_CW = [442.7, 492.4, 559.8, 664.6, 704.1, 740.5, 782.8, 832.8, 864.7, 945.1]

DIRECTIONS = [(-1.0, 0.0), (0.0, -1.0), (0.0, 1.0), (1.0, 0.0)]

RAND_MEAN = 0.0
RAND_STD = 0.04
RELATIVE_SHIFT = 0.204


class ProcessingLevels(Enum):
    L1B = "L1B"
    L1C = "L1C"

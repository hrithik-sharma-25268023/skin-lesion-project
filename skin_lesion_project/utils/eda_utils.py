"""Utils for EDA"""

from enum import Enum

LABEL_COLS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
PALETTE = ["#E24B4A","#378ADD","#1D9E75","#BA7517","#534AB7","#D85A30","#D4537E","#639922"]


class CancerLabels(Enum):
    """Enum for Labels"""

    MEL = "Melanoma"
    NV = "Melanocytic Nevi"
    BCC = "Basal Cell Carcinoma"
    AK = "Actinic Keratosis"
    BKL = "Benign Keratosis"
    DF = "Dermatofibroma"
    VASC = "Vascular Lesion"
    SCC = "Squamous Cell Carcinoma"

CLASS_NAMES = dict(zip(LABEL_COLS, [label.value for label in CancerLabels]))

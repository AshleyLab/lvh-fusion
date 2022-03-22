"""
The hha.datasets submodule defines a Pytorch dataset for loading
ECG signals.
"""

from .ecg import Ecg
from .ecg_labels import EcgLabels
from .ecg_ablation import EcgAblation

__all__ = ["Ecg", "EcgLabels"]
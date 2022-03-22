"""
The hha.losses submodule defines the loss functions for HHA 
"""
from .constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE

from .focal import FocalLoss

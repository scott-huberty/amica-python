from importlib.metadata import PackageNotFoundError, version

from . import datasets, utils
from ._sklearn_interface import AMICA
from .core import fit_amica

try:
    __version__ = version("amica-python")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ['fit_amica', 'AMICA', 'datasets', 'utils', '__version__']

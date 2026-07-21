"""A tool to fit mechanistic models to deep mutational scanning data"""
# Import classes users will interact with

from importlib.metadata import PackageNotFoundError, version

from pymochi.project import MochiProject


try:
    __version__ = version("pymochi")
except PackageNotFoundError:
    __version__ = "0+unknown"

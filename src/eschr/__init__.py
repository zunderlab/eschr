from importlib.metadata import version

from . import pl, readwrite, tl

__all__ = ["pl", "tl", "readwrite"]

__version__ = version("eschr")

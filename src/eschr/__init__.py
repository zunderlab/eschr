from importlib.metadata import version

from . import pl, tl, readwrite

__all__ = ["pl", "tl", "readwrite"]

__version__ = version("eschr")

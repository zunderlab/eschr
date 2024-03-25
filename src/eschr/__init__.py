from importlib.metadata import version

from .read_write_utils import csv_to_zarr, make_zarr
from .plotting import smm_heatmap, umap
from . import pl
from . import tl


#from __future__ import absolute_import
#import eschr._read_write_utils
#import eschr.consensus_cluster
#import eschr.pl

#__all__ = ["pl", "tl", "csv_to_zarr", "make_zarr"]

#from .version import __version__
__version__ = version("eschr")

from importlib.metadata import version

#from . import pl
from .read_write_utils import csv_to_zarr, make_zarr
from .consensus_cluster import ConsensusCluster
from .smm_heatmap import make_smm_heatmap
from .umap import plot_umap

#from __future__ import absolute_import
#import eschr._read_write_utils
#import eschr.consensus_cluster
#import eschr.pl

__all__ = ["ConsensusCluster", "csv_to_zarr", "make_zarr","make_smm_heatmap","plot_umap"]

#from .version import __version__
__version__ = version("eschr")

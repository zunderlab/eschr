from importlib.metadata import version

from .consensus_cluster import ConsensusCluster
# from . import pl
from .read_write_utils import csv_to_zarr, make_zarr
from .plotting import make_smm_heatmap, plot_umap

# from __future__ import absolute_import
# import eschr._read_write_utils
# import eschr.consensus_cluster
# import eschr.pl

__all__ = [
    "ConsensusCluster",
    "csv_to_zarr",
    "make_zarr",
    "make_smm_heatmap",
    "plot_umap",
]

# from .version import __version__
__version__ = version("eschr")

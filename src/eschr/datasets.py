# Adapted from: https://github.com/theislab/cellrank.git
# and https://github.com/LouisFaure/scFates.git

import os
import pathlib
from typing import Any, Literal, Tuple, Union

from anndata import AnnData
from scanpy import read

__all__ = [
    "pancreas",
    "lung",
    "reprogramming_schiebinger",
    "zebrafish",
    "bone_marrow",
]

_datasets = {
    "pancreas": ("https://figshare.com/ndownloader/files/25060877", (2531, 27998)),
    "pancreas_preprocessed": ("https://figshare.com/ndownloader/files/25030028", (2531, 2000)),
    "pancreas_preprocessed_vk": ("https://figshare.com/ndownloader/files/41325411", (2531, 5974)),
    "lung": ("https://figshare.com/ndownloader/files/25038224", (24882, 24051)),
    "zebrafish": ("https://figshare.com/ndownloader/files/27265280", (2434, 23974)),
    "reprogramming_schiebinger": ("https://figshare.com/ndownloader/files/28618734", (236285, 19089)),
    "reprogramming_schiebinger_serum_subset": ("https://figshare.com/ndownloader/files/35858033", (165892, 19089)),
    "bone_marrow": ("https://figshare.com/ndownloader/files/35826944", (5780, 27876)),
}


def _load_dataset_from_url(
    fpath: Union[str, pathlib.Path], url: str, expected_shape: Tuple[int, int], **kwargs: Any
) -> AnnData:
    fpath = str(fpath)
    if not fpath.endswith(".h5ad"):
        fpath += ".h5ad"

    if os.path.isfile(fpath):
        print(f"Loading dataset from `{fpath!r}`")
    else:
        print(f"Downloading dataset from `{url!r}` as `{fpath!r}`")

    dirname, _ = os.path.split(fpath)
    try:
        if not os.path.isdir(dirname):
            print(f"Creating directory `{dirname!r}`")
            os.makedirs(dirname, exist_ok=True)
    except OSError as e:
        raise Exception(f"Unable to create directory `{dirname!r}`. Reason `{e}`")

    kwargs.setdefault("sparse", True)
    kwargs.setdefault("cache", True)

    adata = read(fpath, backup_url=url, **kwargs)

    if adata.shape != expected_shape:
        raise ValueError(f"Expected `anndata.AnnData` object to have shape `{expected_shape}`, found `{adata.shape}`.")

    adata.var_names_make_unique()

    return adata


def pancreas(
    path: Union[str, pathlib.Path] = "datasets/endocrinogenesis_day15.5.h5ad",
    kind: Literal["raw", "preprocessed", "preprocessed-kernel"] = "raw",
    **kwargs: Any,
) -> AnnData:  # pragma: no cover
    """Development of the murine pancreas at E15.5 from :cite:`bastidas-ponce:19`.

    scRNA-seq dataset containing 2531 cells recorded using 10x Chromium in a single time point.
    Data was filtered to remove heavily cycling populations and to focus on the late stages of endocrinogenesis.

    Parameters
    ----------
    path
        Path where to save the dataset.
    kind
        What kind of dataset to download. Valid option are:

        - ``'raw'`` - contains raw spliced and unspliced count data, low-dimensional embedding coordinates
          as well as original cluster annotations.
        - ``'preprocessed'`` - same as above, but additionally genes have been filtered to contain at least
          20 unspliced and spliced counts and subsetted to the top 2000 highly variable genes. Data has been
          normalized by total counts and log-transformed.
        - ``'preprocessed-kernel'`` - same as above, but additionally a cell-cell transition matrix has been computed
          using the :class:`~cellrank.kernels.VelocityKernel`.
    kwargs
        Keyword arguments for :func:`~scanpy.read`.

    Returns
    -------
    Annotated data object.
    """
    path, ext = os.path.splitext(path)
    path = f"{path}_{kind}{ext}"

    if kind == "raw":
        return _load_dataset_from_url(path, *_datasets["pancreas"], **kwargs)
    if kind == "preprocessed":
        return _load_dataset_from_url(path, *_datasets["pancreas_preprocessed"], **kwargs)
    if kind == "preprocessed-kernel":
        return _load_dataset_from_url(path, *_datasets["pancreas_preprocessed_vk"], **kwargs)
    raise ValueError(f"Unknown dataset kind `{kind!r}`.")


def lung(
    path: Union[str, pathlib.Path] = "datasets/lung_regeneration.h5ad",
    **kwargs: Any,
) -> AnnData:  # pragma: no cover
    """Regeneration of murine lung epithelial cells at 13 time points from :cite:`strunz:20`.

    sc-RNA-seq dataset comprising `24 051` cells recorded using Drop-seq :cite:`macosko:15` at 13 time points spanning
    days 2-15 past lung bleomycin injury. Data was filtered to remove control cells as well as later time points which
    are more spaced out. We wanted to focus on the densely sampled days when RNA velocity :cite:`manno:18,bergen:20`
    can be used to predict the future cellular state.

    Contains raw spliced and unspliced count data, low-dimensional embedding coordinates as well as original
    cluster annotations.

    Parameters
    ----------
    path
        Path where to save the dataset.
    kwargs
        Keyword arguments for :func:`~scanpy.read`.

    Returns
    -------
    Annotated data object.
    """
    return _load_dataset_from_url(path, *_datasets["lung"], **kwargs)


def reprogramming_schiebinger(
    path: Union[str, pathlib.Path] = "datasets/reprogramming_schiebinger.h5ad",
    subset_to_serum: bool = False,
    **kwargs: Any,
) -> AnnData:  # pragma: no cover
    """Reprogramming of mouse embryonic fibroblasts to induced pluripotent stem cells from :cite:`schiebinger:19`.

    sc-RNA-seq dataset comprising `236 285` cell recorded using 10X Chromium
    at 39 time points spanning days 0-18 past reprogramming initiation.

    Contains total-counts normalized, log-transformed counts and low-dimensional embedding coordinates (force-directed).
    Moreover, it contains the following :attr:`~anndata.AnnData.obs` annotations:

    - ``'day'`` - time-point information.
    - ``'serum'``/``'2i'`` - whether this cell comes from the serum/2i condition.
    - ``'cell_sets'`` - cluster labels.

    Parameters
    ----------
    path
        Path where to save the dataset.
    subset_to_serum
        Whether to return the full object or subsetted to the serum condition.
        This subset also contains the pre-computed transition matrix.
    kwargs
        Keyword arguments for :func:`~scanpy.read`.

    Returns
    -------
    Annotated data object.

    Notes
    -----
    The full dataset has approximately 1.4GiB.
    """
    if subset_to_serum:
        key = "reprogramming_schiebinger_serum_subset"
        path, ext = os.path.splitext(path)
        path = f"{path}_serum{ext}"
    else:
        key = "reprogramming_schiebinger"

    return _load_dataset_from_url(path, *_datasets[key], **kwargs)


@d.dedent
def zebrafish(
    path: Union[str, pathlib.Path] = "datasets/zebrafish.h5ad",
    **kwargs: Any,
) -> AnnData:  # pragma: no cover
    """Zebrafish embryogenesis assayed using drop-seq from :cite:`farrell:18`.

    sc-RNA-seq time-series dataset, restricted to the axial mesoderm lineage, comprising of `2434` cells which
    contains 12 time-points spanning 3.3-12 hours past fertilization.

    Parameters
    ----------
    path
        Path where to save the dataset.
    kwargs
        Keyword arguments for :func:`~scanpy.read`.

    Returns
    -------
    Annotated data object.
    """
    return _load_dataset_from_url(path, *_datasets["zebrafish"], **kwargs)


@d.dedent
def bone_marrow(
    path: Union[str, pathlib.Path] = "datasets/bone_marrow.h5ad",
    **kwargs: Any,
) -> AnnData:  # pragma: no cover
    """sc-RNA-seq dataset early human hematopoiesis (CD34+ bone marrow cells) assayed using 10X Chromium.

    This dataset contains raw spliced and unspliced counts estimated using *velocyto* :cite:`manno:18`.
    Furthermore, the dataset also contains a precomputed *Palantir* pseudotime :cite:`setty:19`.

    Parameters
    ----------
    path
        Path where to save the dataset.
    kwargs
        Keyword arguments for :func:`~scanpy.read`.

    Returns
    -------
    Annotated data object.
    """
    return _load_dataset_from_url(path, *_datasets["bone_marrow"], **kwargs)

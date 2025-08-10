## Import packages=============================================================
import math
import multiprocessing
import os
import random
import time
import traceback
import warnings
from itertools import repeat

import numpy as np
import pandas as pd
import zarr
from scipy.sparse import coo_matrix, csr_matrix, hstack
from scipy.spatial.distance import pdist, squareform
from sklearn import metrics

from ._zarr_utils import (make_zarr_sparse, make_zarr_dense)
from ._clustering import (run_base_clustering, consensus_cluster_leiden)
from ._prune_features import (  # ADD BACK PRECEDING DOTS
    calc_highly_variable_genes, calc_pca)

## End Import packages section=================================================

# flake8: noqa: B902
# flake8: noqa: E266

## Suppress warnings from printing
warnings.filterwarnings("ignore")
# warnings.filterwarnings('ignore', message='*Note that scikit-learn's randomized PCA might not be exactly reproducible*')


## FUNCTION AND CLASS DOCUMENTATION!!


############################################################################### UTILS
############## Adapted from ......... scedar github
def _parmap_fun(f, q_in, q_out):
    """
    Map function to process.

    Parameters
    ----------
    f : `function`
        Function to run in a given single process.
    q_in :
        Input `multiprocessing.Queue`.
    q_out :
        Output `multiprocessing.Queue`.
    """
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))


def parmap(f, X, nprocs=1):
    """
    Run functions with mutiprocessing.

    parmap_fun() and parmap() are adapted from klaus se's post
    on stackoverflow. https://stackoverflow.com/a/16071616/4638182

    parmap allows map on lambda and class static functions.

    Fall back to serial map when nprocs=1.

    Parameters
    ----------
    f : `function`
        Function to run in parallel single processes.
    X : list of iterables
        List of generators or other iterables containing args for
        specified function.
    nprocs : int
        Number of parallel processes to run

    Returns
    -------
    subsample_size : int
        The number of data points/instances/cells to sample.
    """
    if nprocs < 1:
        raise ValueError(f"nprocs should be >= 1. nprocs: {nprocs}")

    nprocs = min(int(nprocs), multiprocessing.cpu_count())
    # exception handling f
    # simply ignore all exceptions. If exception occurs in parallel queue, the
    # process with exception will get stuck and not be able to process
    # following requests.

    def ehf(x):
        try:
            res = f(x)
        except Exception as e:
            res = e
        return res

    # fall back on serial
    if nprocs == 1:
        return list(map(ehf, X))
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()
    proc = [
        multiprocessing.Process(target=_parmap_fun, args=(f, q_in, q_out))
        for _ in range(nprocs)
    ]
    for p in proc:
        p.daemon = True
        p.start()
    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]
    [p.join() for p in proc]
    # maintain the order of X
    ordered_res = [x for i, x in sorted(res)]
    for i, x in enumerate(ordered_res):
        if isinstance(x, Exception):
            warnings.warn(f"{x} encountered in parmap {i}th arg {X[i]}")
    return ordered_res


############################################################################### MAIN FUNCTIONS


def ensemble(
    zarr_loc, reduction, metric, ensemble_size, k_range, la_res_range, nprocs, sparse
):
    """
    Run ensemble of clusterings.

    Parameters
    ----------
    zarr_loc : str
        Path to save zarr store which will hold the data to be clustered.
    reduction : {'all', ‘pca’}
        Which method to use for feature extraction, or `all` for use all features. 
        Currently only PCA is supported, but alternative options will be added in 
        future releases.
    metric : {'euclidean', 'cosine', None}
        Metric used for neighborhood graph construction. Can be "euclidean",
        "cosine", or `None`. Default is `None`, in which case the metric is
        randomly selected for each ensemble member. Other metrics will be
        added in future releases.
    ensemble_size : int, default=150
        Number of clusterings to run in the ensemble.
    k_range : tuple of (int, int)
        Upper and lower limits for selecting random k for neighborhood
        graph construction.
    la_res_range : tuple of (int, int)
        Upper and lower limits for selecting random resolution
        parameter for leiden community detection.
    nprocs : int, default=None
        How many processes to run in parallel. If `None`, value is set using
        `multiprocessing.cpu_count()` to find number of available cores.
        This is used as a check and the minimum value between number of
        cores detected and specified number of processes is set as final value.
    sparse : bool, default=None
        Whether the zarr store contains a sparse matrix or not.

    Returns
    -------
    clust_out : :class:`~scipy.sparse.csr_matrix`
        Adjacency matrix generated from the ensemble of clusterings, where
        data points have edges to clusters they were assigned to in
        each ensemble member where they were included.
    """
    start_time = time.perf_counter()

    data_iterator = repeat(zarr_loc, ensemble_size)
    hyperparam_iterator = [
        [k_range, la_res_range, metric] for x in range(ensemble_size)
    ]
    sparse_iterator = repeat(sparse, ensemble_size)
    reduction_iterator = repeat(reduction, ensemble_size)
    args = list(zip(data_iterator, hyperparam_iterator, sparse_iterator, reduction_iterator))

    print("Starting ensemble clustering multiprocess")
    out = parmap(run_base_clustering, args, nprocs=nprocs)

    try:
        clust_out = hstack(out)  
    except Exception:
        print(
            "consensus_cluster.py, line 599, in ensemble: clust_out = hstack(out[:,0])"
        )

    finish_time = time.perf_counter()
    print(f"Ensemble clustering finished in {finish_time-start_time} seconds")

    return clust_out


def consensus(n, bg, nprocs):
    """
    Find consensus from ensemble of clusterings.

    Parameters
    ----------
    n : int
        Number of data points/instances/cells.
    bg :  :class:`~scipy.sparse.csr_matrix`
        The adjacency matrix to build the bipartite graph generated
        from the ensemble of clusterings.
    nprocs : int, default=None
        How many processes to run in parallel. If `None`, value is set using
        `multiprocessing.cpu_count()` to find number of available cores.
        This is used as a check and the minimum value between number of
        cores detected and specified number of processes is set as final value.

    Returns
    -------
    hard_clusters : :class:`numpy.ndarray`
        Final hard cluster assignments for every sample.
    soft_membership_matrix : :class:`numpy.ndarray`
        Contains membership values for each sample in each final consensus
        cluster.
    all_clusterings_df : :class:`pandas.DataFrame`
        Contains hard cluster assignments for each sample in each resolution
        tested for final consensus clustering.
    """
    ## Run final consensus
    res_ls = [x / 1000 for x in range(50, 975, 25)]  # 0.05 to 0.95 inclusive by 0.025

    print("Starting consensus multiprocess")
    start_time = time.perf_counter()
    bg_iterator = repeat(bg, len(res_ls))
    n_iterator = repeat(n, len(res_ls))
    args = list(zip(n_iterator, res_ls, bg_iterator))
    out = parmap(consensus_cluster_leiden, args, nprocs=nprocs)

    all_clusterings = [pd.DataFrame(x[0], dtype=int) for x in out]
    all_clusterings_df = pd.concat(all_clusterings, axis=1)
    all_clusterings_df.columns = list(range(all_clusterings_df.shape[1]))

    # find res with minimum sum of distances to all other res
    dist = pdist(
        all_clusterings_df.T, metric=lambda u, v: 1 - metrics.adjusted_rand_score(u, v)
    )
    opt_res_idx = np.argmin(squareform(dist).sum(axis=0))

    # extract final hard and soft clusters for selected optimal resolution
    hard_clusters = out[opt_res_idx][0]  # kl.knee
    soft_membership_matrix = out[opt_res_idx][1].toarray()
    print("Final res: " + str(out[opt_res_idx][2]))

    finish_time = time.perf_counter()
    print(f"Consensus clustering finished in {finish_time-start_time} seconds")

    return hard_clusters, soft_membership_matrix, all_clusterings_df.to_numpy(dtype=np.uint16) 


def consensus_cluster(
    adata,
    zarr_loc,
    reduction="pca",
    metric=None,  # how to add options?
    ensemble_size=150,
    auto_stop=False,
    k_range=(15, 150),
    la_res_range=(25, 175),
    nprocs=None,
    return_multires=False,
):
    """
    Run ensemble of clusterings and find consensus.

    Runs ensemble of leiden clusterings on random subsamples of input data with
    random hyperparameters from within the default range (or user specified).
    Then generates a bipartite graph from these results where data instances
    have edges to all clusters they were assigned to accross the ensemble of
    clsuterings. Bipartite community detection is run on this resulting graph
    to obtain final hard and soft clusters.

    Parameters
    ----------
    adata : `anndata.AnnData`
        AnnData object containing preprocessed data to be clustered in slot `.X`
    zarr_loc : str
        Path to save zarr store which will hold the data to be clustered.
    reduction : {'all', ‘pca’}
        Which method to use for feature extraction/selection/dimensionality
        reduction, or `all` for use all features. Currently only PCA is
        supported, but alternative options will be added in future releases.
        Once other options are added, the default will be to randomly select
        a reduction for each ensemble member. For datasets with fewer than
        10 features, all features are used.
    metric : {'euclidean', 'cosine', None}
        Metric used for neighborhood graph construction. Can be "euclidean",
        "cosine", or `None`. Default is `None`, in which case the metric is
        randomly selected for each ensemble member. Other metrics will be
        added in future releases.
    ensemble_size : int, default=150
        Number of clusterings to run in the ensemble.
    k_range : tuple of (int, int)
        Upper and lower limits for selecting random k for neighborhood
        graph construction.
    la_res_range : tuple of (int, int)
        Upper and lower limits for selecting random resolution
        parameter for leiden community detection.
    nprocs : int, default=None
        How many processes to run in parallel. If `None`, value is set using
        `multiprocessing.cpu_count()` to find number of available cores.
        This is used as a check and the minimum value between number of
        cores detected and specified number of processes is set as final value.
    return_multires : bool, default=False
        Whether or not to add consensus results from all tested resolutions to the
        adata object. Default is `False` as this can add subtantial memory usage.

    Returns
    -------
    `anndata.AnnData` object modified in place.
    """
    start_time = time.time()

    # Set number of processes
    if nprocs is None:
        nprocs = multiprocessing.cpu_count()
    nprocs = min(int(nprocs), multiprocessing.cpu_count())
    print("Multiprocessing will use " + str(nprocs) + " cores")

    # Test sparseness
    if (isinstance(adata.X, csr_matrix)) or (isinstance(adata.X, coo_matrix)):
        sparse = True
    else:
        sparsity = 1.0 - np.count_nonzero(adata.X) / adata.X.size
        if sparsity > 0.1:
            sparse = True
        else:
            sparse = False
    # Make zarr store for multiproces data access
    # NEED TO ADD SOME SORT OF CHECK FOR THIS STEP
    # to test if path is valid, if data is actually there
    if os.path.exists(zarr_loc) == False:
        print("making zarr")
        if sparse:
            make_zarr_sparse(adata, zarr_loc)
        else:
            make_zarr_dense(adata, zarr_loc)

    # Generate ensemble of base clusterings
    k_range = (int(k_range[0]), int(k_range[1]))
    la_res_range = (
        int(la_res_range[0]),
        int(la_res_range[1]),
    )  
    bipartite = ensemble(
        zarr_loc=zarr_loc,
        reduction=reduction,
        metric=metric,
        ensemble_size=ensemble_size,
        k_range=k_range,
        la_res_range=la_res_range,
        nprocs=nprocs,
        sparse=sparse,
    )

    # Obtain consensus from ensemble
    hard_clusters, soft_membership_matrix, all_clusterings = consensus(
        n=bipartite.shape[0], bg=bipartite, nprocs=nprocs
    )

    print("Final Clustering:")
    print("n hard clusters: " + str(len(np.unique(hard_clusters))))
    print("n soft clusters: " + str(soft_membership_matrix.shape[1]))

    ## Add results to adata object
    adata.obs["hard_clusters"] = pd.Categorical(hard_clusters)
    adata.obsm["soft_membership_matrix"] = soft_membership_matrix
    adata.obs["uncertainty_score"] = 1 - np.max(soft_membership_matrix, axis=1)
    adata.obsm["bipartite"] = bipartite.tocsr()
    if return_multires:
        adata.obsm["multiresolution_clusters"] = all_clusterings

    time_per_iter = time.time() - start_time
    print("Full runtime: " + str(time_per_iter))

    return adata

## Import packages=============================================================
import math
import multiprocessing
import os
import random
import time
import traceback
import warnings
from itertools import repeat

import leidenalg as la
import numpy as np
import pandas as pd
import zarr
from igraph import Graph
from scipy.sparse import coo_matrix, csr_matrix, hstack
from scipy.spatial.distance import pdist, squareform
from sklearn import metrics

from ._leiden import run_la_clustering  # _base_clustering_utils _leiden
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


def make_zarr(adata, zarr_loc):
    """
    Make zarr data store.

    Parameters
    ----------
    adata : `anndata.AnnData`
        AnnData object containing preprocessed data to be clustered in slot `.X`
    zarr_loc : str
        Path to save zarr store which will hold the data to be clustered.
    """
    if zarr_loc == None:
        zarr_loc = os.getcwd() + "/data_store.zarr"
    print("storing zarr data object as " + zarr_loc)
    data = coo_matrix(adata.X)
    z1 = zarr.open(zarr_loc, mode="w")
    X = z1.create_group("X")
    data_row = X.create_dataset(
        name="row", shape=data.row.shape, chunks=False, dtype="int32", overwrite=True
    )
    data_row[:] = data.row
    data_col = X.create_dataset(
        name="col", shape=data.col.shape, chunks=False, dtype="int32", overwrite=True
    )
    data_col[:] = data.col
    data_data = X.create_dataset(
        name="data",
        shape=data.data.shape,
        chunks=False,
        dtype="float32",
        overwrite=True,
    )
    data_data[:] = data.data


############################################################################### UTILS
def get_subsamp_size(n):  # n==data.shape[0]
    """
    Generate subsample size.

    Calculates subsample size for a single clustering.
    Value is chosen from a gaussian whose center is set
    based on the number of data points/instances/cells
    in the dataset.

    Parameters
    ----------
    n : int
        Number of data points/instances/cells.

    Returns
    -------
    subsample_size : int
        The number of data points/instances/cells to sample.
    """
    oom = math.ceil(n / 1000)
    # print(oom)
    if oom > 1000:  # aka more than 1 mil data points
        mu = 30
    elif oom == 1:  # aka fewer than 1000 data points
        mu = 90
    else:
        oom = 1000 - oom  # so that it scales in the appropriate direction
        mu = ((oom - 1) / (1000 - 1)) * (90 - 30) + 30
    subsample_ratio = random.gauss(mu=mu, sigma=10)
    while subsample_ratio >= 100 or subsample_ratio < 10:
        subsample_ratio = random.gauss(mu=mu, sigma=10)
    ## Calculate subsample size
    subsample_size = math.ceil((subsample_ratio / 100) * n)
    return subsample_size


## Get hyperparameters
def get_hyperparameters(k_range, la_res_range, metric=None):
    """
    Calculate hyperparameters for a single clustering.

    Parameters
    ----------
    k_range : tuple of (int, int)
        Upper and lower limits for selecting random k for neighborhood
        graph construction.
    la_res_range : tuple of (int, int)
        Upper and lower limits for selecting random resolution
        parameter for leiden community detection.
    metric : {‘euclidean’, ‘cosine’, None}, optional
        Which distance metric to use for calculating kNN graph for clustering.
        For now, one of ('cosine', 'euclidean'), plan to add
        correlation when I can find a fast enough implementation.

    Returns
    -------
    k : int
        Number of neighbors for neighborhood graph construction.
    la_res : int
        Resolution parameter for leiden community detection.
    metric : str
        Which distance metric to use for calculating kNN graph for clustering.
        For now, one of ('cosine', 'euclidean'), plan to add
        correlation when I can find a fast enough implementation.
    """
    k = random.sample(range(k_range[0], k_range[1]), 1)[0]
    la_res = random.sample(range(la_res_range[0], la_res_range[1]), 1)[0]
    if metric is None:
        metric = ["euclidean", "cosine"][random.sample(range(2), 1)[0]]
    return k, la_res, metric


def run_pca_dim_reduction(X):
    """
    Produce PCA-reduced data matrix.

    Generates a dimensionality-reduced data matrix through
    PCA feature extraction. Other methods of feature extraction
    and selection will be included in future releases.

    Parameters
    ----------
    X : :class:`~numpy.array` or :class:`~scipy.sparse.spmatrix`
        Data matrix of shape `n_obs` × `n_vars`. Rows correspond
        to cells (or other instance type) and columns to genes (or other feature type).

    Returns
    -------
    X_pca : :class:`~numpy.array` or :class:`~scipy.sparse.spmatrix`
        Data matrix of shape `n_obs` × `n_pcs`. Rows correspond
        to cells and columns to PCA-extracted features.
    """
    time.time()
    if X.shape[1] > 6000:  # somewhat arbitrary cutoff, come up with better heuristic?
        bool_features = calc_highly_variable_genes(X)
        X = X[:, bool_features]
    X_pca = np.array(calc_pca(X))
    return X_pca


def run_base_clustering(args_in):
    """
    Run a single iteration of leiden clustering.

    Parameters
    ----------
    args_in : zip
        List containing each hyperparameter required for one round of
        clustering (k, la_res, metric, subsample_size) as well as a
        copy of the data as numpy.array or scipy.sparse.spmatrix

    Returns
    -------
    coo_matrix(c).tocsr() : :class:`~scipy.sparse.spmatrix`
        Matrix of dimensions n (total number of data points) by
        m (number of clusters) and filled with 1/0 binary occupancy
        of data point per cluster.
    """
    try:
        # LOAD DATA
        zarr_loc = args_in[0]
        hyperparams_ls = args_in[1]

        z1 = zarr.open(zarr_loc, mode="r")
        data = coo_matrix(
            (z1["X"]["data"][:], (z1["X"]["row"][:], z1["X"]["col"][:])),
            shape=[np.max(z1["X"]["row"][:]) + 1, np.max(z1["X"]["col"][:]) + 1],
        ).tocsr()

        # Calculate subsample size for this ensemble member
        subsample_size = get_subsamp_size(data.shape[0])
        # Get indices for random subsample
        subsample_ids = random.sample(range(data.shape[0]), subsample_size)
        ## Subsample data
        n_orig = data.shape[0]  # save original number of data points
        data = data[subsample_ids, :]

        # Get hyperparameters
        # scale k range for selecting number of neighbors
        # based on size of subsampled data
        i = data.shape[0]
        # set iter_k range floor based on asymptotic function of cell number, lower bound of 2
        k_floor = max(2, ((16 * i) / (1 * (i) + 6000)))
        # set iter_k range ceiling based on dataset size
        # min ceiling is 5, otherwise based on asymptotic function of cell number
        k_ceil = max(5, (160 * i) / (1 * (i) + 6000))
        iter_k_range = (int(k_floor), int(k_ceil))

        # Get hyperparameter settings for this ensemble member
        iter_k, la_res, metric = get_hyperparameters(
            iter_k_range, hyperparams_ls[1], hyperparams_ls[2]
        )

        ## internal heuristic for if it scrna seq and not log transformed
        if data.shape[1] > 8000:
            if np.max(data) > 20:
                print("Data likely needs to be preprocessed, results may be suboptimal")

        ## Data subspace feature extraction
        data = run_pca_dim_reduction(data)
        ## Run leiden clustering
        clusters = run_la_clustering(
            X=data, k=iter_k, la_res=la_res / 100, metric=metric
        )
        ## Prepare outputs for this ensemble member
        len(np.unique(clusters))
        a = np.zeros((n_orig), dtype=np.uint8)
        a[subsample_ids] = clusters[0] + 1
        b = np.ones((n_orig), dtype=np.uint8)
        c = np.zeros((n_orig, len(np.unique(a))), dtype=np.uint8)
        np.put_along_axis(
            arr=c,
            indices=np.expand_dims(a, axis=1),
            values=np.expand_dims(b, axis=1),
            axis=1,
        )  # )#,
        c = np.delete(c, 0, 1)

    except Exception as ex:
        traceback.print_exception(type(ex), ex, ex.__traceback__)
        return ["error", data]

    return coo_matrix(c)


def get_hard_soft_clusters(n, clustering, bg):
    """
    Generate hard and soft clusters for a single bipartite clustering.

    Parameters
    ----------
    n : int
        Number of data points/instances/cells.
    clustering : ndarray of shape (n_cons_clust, 1)
        Consensus cluster assignment for each ensemble cluster.
    bg :  :class:`~igraph.Graph`
        The bipartite graph generated from the ensemble of clusterings.

    Returns
    -------
    hard_clusters : int
        Hard cluster assignments for every sample.
    soft_membership_matrix : :class:`numpy.ndarray`
        Contains mebership values for each sample in each consensus cluster.
    """
    clusters_vertex_ids = np.array(bg.vs.indices)[[x >= n for x in bg.vs.indices]]
    cells_clusts = np.unique(clustering)
    clust_occ_arr = np.zeros((n, len(cells_clusts)), int)
    for v in range(len(cells_clusts)):
        cluster_id = cells_clusts[v]
        cluster_memb = [
            clusters_vertex_ids[i] for i, j in enumerate(clustering) if j == cluster_id
        ]
        node_subset, counts = np.unique(
            [e.source for e in bg.es.select(_source_in=cluster_memb)],
            return_counts=True,
        )
        clust_occ_arr[node_subset, v] = counts
    hard_clusters = np.array(
        [np.random.choice(np.where(row == row.max())[0]) for row in clust_occ_arr]
    )
    soft_membership_matrix = clust_occ_arr / clust_occ_arr.sum(axis=1, keepdims=True)
    return hard_clusters, soft_membership_matrix


def consensus_cluster_leiden(in_args):
    """
    Runs a single iteration of leiden clustering.

    Parameters
    ----------
    in_args : zip
        List containing (1) the number of data points, (2) the bipartite
        leiden clustering resolution, and (3) the bipartite graph generated
        from the ensemble of clusterings.

    Returns
    -------
    hard_clusters :  :class:`~pandas.Series`
        Categorical series containing hard cluster assignments per data point.
    csr_matrix(soft_membership_matrix) : :class:`~scipy.sparse.spmatrix`
        Matrix of dimensions n (total number of data points) by
        m (number of consensus clusters) and filled with membership ratio
        of data point per cluster.
    i : float
        Bipartite leiden resolution parameter, a sanity check to ensure
        parallel processing maintains expected order of resolutions
        in output.
    """
    ## Run initial Lieden clustering with specified resolution value
    n = in_args[0]
    i = in_args[1]
    # Make bipartite igraph from sparse matrix
    bipartite = Graph(
        np.concatenate(
            (
                np.expand_dims(in_args[2].row, axis=1),
                np.expand_dims(in_args[2].col + n, axis=1),
            ),
            axis=1,
        )
    ).as_undirected()
    type_ls = [0] * n
    type_ls.extend([1] * (bipartite.vcount() - n))
    bipartite.vs["type"] = type_ls
    assert bipartite.is_bipartite()
    p_01, p_0, p_1 = la.CPMVertexPartition.Bipartite(
        bipartite, resolution_parameter_01=i
    )
    optimiser = la.Optimiser()
    diff = optimiser.optimise_partition_multiplex(
        partitions=[p_01, p_0, p_1], layer_weights=[1, -1, -1]
    )
    clustering = np.array(p_01.membership)[
        np.where(bipartite.vs["type"])[0]
    ]  # just select clusters assigns for clusters
    clustering_cells = np.array(p_01.membership)[
        [i for i, val in enumerate(bipartite.vs["type"]) if not val]
    ]  # just select clusters assigns for cells?
    hard_clusters, soft_membership_matrix = get_hard_soft_clusters(
        n, clustering, bipartite
    )
    # convert resulting membership back to ratio*
    soft_membership_matrix = np.divide(
        soft_membership_matrix, soft_membership_matrix.sum(axis=1)[:, None]
    )
    # calculate final hard clusters based on majority vote by membership
    hard_clusters = pd.Categorical(
        np.array(
            [
                np.random.choice(np.where(row == row.max())[0])
                for row in soft_membership_matrix
            ]
        )
    )

    return hard_clusters, csr_matrix(soft_membership_matrix), i  # , ari


############################################################################### MAIN FUNCTIONS


def ensemble(zarr_loc, reduction, metric, ensemble_size, k_range, la_res_range, nprocs):
    """
    Run ensemble of clusterings.

    Parameters
    ----------
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
    args = list(zip(data_iterator, hyperparam_iterator))

    print("starting ensemble clustering multiprocess")
    # out = np.array(parmap(run_base_clustering, args, nprocs=nprocs))
    out = parmap(run_base_clustering, args, nprocs=nprocs)

    try:
        clust_out = hstack(out)  # [x[0] for x in out]
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

    print("starting consensus multiprocess")
    start_time = time.perf_counter()
    bg_iterator = repeat(bg, len(res_ls))
    n_iterator = repeat(n, len(res_ls))
    args = list(zip(n_iterator, res_ls, bg_iterator))
    # out = np.array(parmap(consensus_cluster_leiden, args, nprocs=self.nprocs))
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

    return hard_clusters, soft_membership_matrix, all_clusterings_df


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

    if nprocs is None:
        nprocs = multiprocessing.cpu_count()
    nprocs = min(int(nprocs), multiprocessing.cpu_count())
    print("Multiprocessing will use " + str(nprocs) + " cores")

    # Make zarr store for multiproces data access
    # NEED TO ADD SOME SORT OF CHECK FOR THIS STEP
    # to test if path is valid, if data is actually there
    if os.path.exists(zarr_loc) == False:
        print("making zarr")
        make_zarr(adata, zarr_loc)

    # Generate ensemble of base clusterings
    k_range = (int(k_range[0]), int(k_range[1]))
    la_res_range = (
        int(la_res_range[0]),
        int(la_res_range[1]),
    )  # , per_iter_clust_assigns
    bipartite = ensemble(
        zarr_loc=zarr_loc,
        reduction=reduction,
        metric=metric,
        ensemble_size=ensemble_size,
        k_range=k_range,
        la_res_range=la_res_range,
        nprocs=nprocs,
    )

    # Obtain consensus from ensemble
    hard_clusters, soft_membership_matrix, all_clusterings_df = consensus(
        n=bipartite.shape[0], bg=bipartite, nprocs=nprocs
    )

    print("Final Clustering:")
    print("n hard clusters: " + str(len(np.unique(hard_clusters))))
    print("n soft clusters: " + str(soft_membership_matrix.shape[1]))

    ## Add results to adata object
    adata.obs["hard_clusters"] = hard_clusters
    adata.obsm["soft_membership_matrix"] = soft_membership_matrix
    adata.obs["uncertainty_score"] = 1 - np.max(soft_membership_matrix, axis=1)
    adata.obsm["bipartite"] = bipartite
    if return_multires:
        adata.obsm["multiresolution_clusters"] = all_clusterings_df

    time_per_iter = time.time() - start_time
    print("Full runtime: " + str(time_per_iter))

    return adata

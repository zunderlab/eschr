import math
import random
import time
import traceback
import warnings

import igraph as ig
import leidenalg as la
import zarr
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix, diags
from sklearn_ann.kneighbors.annoy import AnnoyTransformer

from ._prune_features import run_pca_dim_reduction

warnings.filterwarnings("ignore")

########################################################################################################################################################
# Hyperparameter Utils
########################################################################################################################################################

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

########################################################################################################################################################
# Clustering Utils
########################################################################################################################################################

def sparse_put_clusters(n_orig, subsample_ids, cluster_values):
    """Create a sparse cluster matrix without using put_along_axis"""
    
    # Get number of clusters (accounting for zero as non-cluster)
    n_clusters = len(np.unique(cluster_values))
    
    # Create COO matrix directly from indices and values
    # For each data point in subsample_ids, create a 1 in its cluster column
    rows = subsample_ids
    cols = cluster_values
    data = np.ones_like(subsample_ids, dtype=np.uint8)
    
    # Create the sparse matrix
    c = coo_matrix((data, (rows, cols)), shape=(n_orig, n_clusters))
    
    return c

# Util adapted from scanpy:
def get_igraph_from_adjacency(adjacency, directed=None):
    """Get igraph graph from adjacency matrix."""
    sources, targets = adjacency.nonzero()
    weights = adjacency[sources, targets]
    if isinstance(weights, np.matrix):
        weights = weights.A1
    g = ig.Graph(directed=directed)
    g.add_vertices(adjacency.shape[0])  # this adds adjacency.shape[0] vertices
    g.add_edges(list(zip(sources, targets)))
    try:
        g.es["weight"] = weights
    except KeyError:
        pass
    return g


def run_la_clustering(X, k, la_res, metric="euclidean", method="sw-graph"):
    """
    Find consensus from ensemble of clusterings.

    Parameters
    ----------
    X : :class:`~numpy.array` or :class:`~scipy.sparse.spmatrix`
        Data matrix of shape `n_obs` × `n_vars`. Rows correspond
        to cells and columns to genes.
    k : int
        Number of neighbors for neighborhood graph construction.
    la_res : int
        Resolution parameter for leiden community detection.
    metric : str
        Which distance metric to use for calculating kNN graph for clustering.
    method : str, default 'sw-graph'
        Neighbor search method. Default is Small World Graph. See
        `https://github.com/nmslib/nmslib/blob/master/manual/methods.md`
        for a list of available methods.

    Returns
    -------
    Array of cluster memberships.
    """
    # start_time = time.time()
    if metric == "jaccard":
        len(X)
    else:
        X.shape[0]

    if metric == "cosine":
        metric = "angular"  # this is how annoy calls it

    # Get k nearest neighbors to input for clustering
    transformer = AnnoyTransformer(n_neighbors=k, metric=metric)
    adjacency_sparse = transformer.fit_transform(X)
    # time_leiden = time.time() - start_time
    # print ("time to run nearest neighbors: " + str(time_leiden))

    # Extract info from nearest neighbors and create iGraph object
    knn_graph = get_igraph_from_adjacency(adjacency=adjacency_sparse, directed=None)

    # get Leiden clusters
    # start_time = time.time()
    leiden_out = knn_graph.community_leiden(
        "modularity", weights="weight", resolution=la_res
    )
    # time_leiden = time.time() - start_time
    # print ("time to run leiden clustering: " + str(time_leiden))
    return np.array([leiden_out.membership])

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
    soft_membership_matrix : :class:`scipy.sparse.csr_matrix`
        Contains membership values for each sample in each consensus cluster.
    """    
    # Identify cluster vertices
    clusters_vertex_ids = np.array(bg.vs.indices)[[x >= n for x in bg.vs.indices]]
    # Get unique cluster assignments
    cells_clusts = np.unique(clustering)
    # Create mapping from cluster ID to column index
    clust_id_to_idx = {clust_id: idx for idx, clust_id in enumerate(cells_clusts)}
    
    # Initialize sparse matrix in LIL format (efficient for incremental construction)
    clust_occ_mat = lil_matrix((n, len(cells_clusts)), dtype=int)
    
    # Process each cluster
    for cluster_id in cells_clusts:
        # Get the vertices corresponding to this cluster
        cluster_memb = [
            clusters_vertex_ids[i] for i, j in enumerate(clustering) if j == cluster_id
        ]
        
        # Get the edges from cells to this cluster
        edges = bg.es.select(_source_in=cluster_memb)
        
        if edges:
            # Get the source nodes and their counts
            sources = [e.source for e in edges]
            source_nodes, counts = np.unique(sources, return_counts=True)
            
            # Update the sparse matrix for this cluster
            col_idx = clust_id_to_idx[cluster_id]
            clust_occ_mat[source_nodes, col_idx] = counts
    
    # Convert to CSR format for efficient row operations
    clust_occ_csr = clust_occ_mat.tocsr()
    
    # Find the max value index for each row (for hard assignments)
    row_maxes = []
    hard_clusters = np.zeros(n, dtype=int)
    
    # Process each row to find max value index
    for i in range(n):
        row = clust_occ_csr[i].toarray().flatten()
        if np.any(row > 0):  # Check if row has any non-zero values
            max_indices = np.where(row == row.max())[0]
            hard_clusters[i] = np.random.choice(max_indices)
    
    # Create the soft membership matrix (normalize rows)
    row_sums = clust_occ_csr.sum(axis=1).A.flatten()
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    
    # Create a diagonal matrix with 1/row_sum
    from scipy.sparse import diags
    row_sum_diag_inv = diags(1.0 / row_sums, 0)
    
    # Multiply to normalize rows
    soft_membership_matrix = row_sum_diag_inv @ clust_occ_csr
    
    return hard_clusters, soft_membership_matrix

########################################################################################################################################################
# Main clustering
########################################################################################################################################################

def run_base_clustering(args_in):
    """
    Run a single iteration of leiden clustering.

    Parameters
    ----------
    args_in : zip
        List containing each hyperparameter required for one round of
        clustering (k, la_res, metric, subsample_size) as well as the 
        sparse boolean and the path to the zarr data store.

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
        sparse = args_in[2]
            
        z1 = zarr.open(zarr_loc, mode="r")

        if sparse:
            data_shape = [np.max(z1["X"]["row"][:]) + 1, np.max(z1["X"]["col"][:]) + 1]
        else:
            data_shape = [z1["X"].shape[0], z1["X"].shape[1]]

        # Calculate subsample size for this ensemble member
        subsample_size = get_subsamp_size(data_shape[0])
        # Get indices for random subsample
        subsample_ids = random.sample(range(data_shape[0]), subsample_size)
        ## Subsample data
        n_orig = data_shape[0]  # save original number of data points
        if sparse:
            row_idxs = np.nonzero(np.isin(z1["X"]["row"], subsample_ids))[0]
            data = coo_matrix(
                (
                    z1["X"]["data"][row_idxs],
                    (z1["X"]["row"][row_idxs], z1["X"]["col"][:]),
                ),
                shape=[
                    np.max(z1["X"]["row"][row_idxs]) + 1,
                    np.max(z1["X"]["col"][:]) + 1,
                ],
            ).tocsr()
        else:
            data = z1["X"][subsample_ids, :]

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
        del data
        ## Prepare outputs for this ensemble member
        c = sparse_put_clusters(n_orig, subsample_ids, clusters[0])

    except Exception as ex:
        traceback.print_exception(type(ex), ex, ex.__traceback__)
        return ["error", str(ex)]

    return coo_matrix(c)


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
    bipartite = ig.Graph(
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

    return hard_clusters, soft_membership_matrix, i

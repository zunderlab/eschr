import warnings

import igraph as ig
# import leidenalg as la
import numpy as np
from scipy.sparse import coo_matrix
from sklearn_ann.kneighbors.annoy import AnnoyTransformer

from ._prune_features import run_pca_dim_reduction

warnings.filterwarnings("ignore")
########################################################################################################################################################
# Clustering Utils
########################################################################################################################################################


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
        sparse = hyperparams_ls[3]

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
        print(np.unique(clusters))
        print(np.unique(a))
        return ["error", data]

    return coo_matrix(c)

import warnings

import igraph as ig
# import leidenalg as la
import numpy as np
from scipy.sparse import coo_matrix
from sklearn_ann.kneighbors.annoy import AnnoyTransformer

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


# Util adapted from scanpy:
def _get_sparse_matrix_from_indices_distances_umap(
    knn_indices, knn_dists, n_obs, n_neighbors
):
    rows = np.zeros((n_obs * knn_indices.shape[1]), dtype=np.int64)  # n_neighbors
    cols = np.zeros((n_obs * knn_indices.shape[1]), dtype=np.int64)  # n_neighbors
    vals = np.zeros((n_obs * knn_indices.shape[1]), dtype=np.float64)  # n_neighbors

    for i in range(knn_indices.shape[0]):
        for j in range(knn_indices.shape[1]):  # n_neighbors
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            if knn_indices[i, j] == i:
                val = 0.0
            else:
                val = knn_dists[i, j]

            rows[i * knn_indices.shape[1] + j] = i  # n_neighbors
            cols[i * knn_indices.shape[1] + j] = knn_indices[i, j]  # n_neighbors
            vals[i * knn_indices.shape[1] + j] = val  # n_neighbors
    result = coo_matrix((vals, (rows, cols)), shape=(n_obs, n_obs))
    result.eliminate_zeros()
    return result.tocsr()


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

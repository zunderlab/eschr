import os
import random
import shutil

import anndata
import numpy as np
import pandas as pd
import pytest
import zarr
from igraph import Graph
from scipy.sparse import coo_matrix, csr_matrix

import eschr as es

# to run test_eschr.py on your local machine, please set up as follows:
# - install extra package into your eschr environment: pytest, pytest-cov
# - install editable version of eschr package (pip install -e .)
#      - check version (conda list eschr), make sure it is the development version
# - make sure the test data are ready under "data" folder


def test_package_has_version():
    es.__version__


@pytest.mark.skip(reason="This decorator should be removed when test passes.")
def test_example():
    assert 1 == 0  # This test is designed to fail.


@pytest.fixture
def X():
    return np.random.rand(100, 1000)


@pytest.fixture
def adata():
    data = pd.read_csv("data/test_data.csv", index_col=0).to_numpy()
    adata = anndata.AnnData(X=data)
    return adata


@pytest.fixture
def adata_with_results():
    adata = anndata.read_h5ad("data/test_adata_with_eschr_results.h5ad")
    return adata


@pytest.fixture
def zarr_loc():
    return "data/test_data.zarr"


@pytest.fixture
def zarr_loc_static(adata):
    es.tl._zarr_utils.make_zarr_dense(adata, "data/test_data_static.zarr")
    return "data/test_data_static.zarr"


# TEST TOOL FUNCTIONS


# Test make_zarr
def test_make_zarr_default(adata):
    es.tl._zarr_utils.make_zarr_dense(adata, None)
    assert os.path.exists("data_store.zarr")
    shutil.rmtree("data_store.zarr")


def test_make_zarr_custom_path(adata, zarr_loc):
    es.tl._zarr_utils.make_zarr_dense(adata, zarr_loc)
    assert os.path.exists(zarr_loc)
    shutil.rmtree(zarr_loc)

@pytest.mark.skip(reason="Update to be testing zarr dense data structure")
def test_make_zarr_content(adata, zarr_loc):
    es.tl._zarr_utils.make_zarr_dense(adata, zarr_loc)
    z = zarr.open(zarr_loc)
    X = z["X"]
    adata_X_coo = coo_matrix(adata.X)
    assert np.array_equal(X["row"][:], adata_X_coo.row)
    assert np.array_equal(X["col"][:], adata_X_coo.col)
    assert np.allclose(X["data"][:], adata_X_coo.data)  # allow for rounding differences
    shutil.rmtree(zarr_loc)


# Test get_subsamp_size
def test_get_subsamp_size():
    # Test extreme small n
    n = 10
    subsample_size_10 = []
    for i in range(1000):
        subsample_size_10.append(es.tl._clustering.get_subsamp_size(n))
    subsample_frac_10 = np.mean(subsample_size_10) / n

    # Test with small n
    n = 500
    subsample_size_500 = []
    for i in range(1000):
        subsample_size_500.append(es.tl._clustering.get_subsamp_size(n))
    subsample_frac_500 = np.mean(subsample_size_500) / n

    # Test with medium n
    n = 50000
    subsample_size_50k = []
    for i in range(1000):
        subsample_size_50k.append(es.tl._clustering.get_subsamp_size(n))
    subsample_frac_50k = np.mean(subsample_size_50k) / n

    # Test with large n
    n = 1000000
    subsample_size_1mil = []
    for i in range(1000):
        subsample_size_1mil.append(es.tl._clustering.get_subsamp_size(n))
    subsample_frac_1mil = np.mean(subsample_size_1mil) / n

    # Test extreme large n
    n = 100000000
    subsample_size_100mil = []
    for i in range(1000):
        subsample_size_100mil.append(es.tl._clustering.get_subsamp_size(n))
    subsample_frac_100mil = np.mean(subsample_size_100mil) / n

    assert subsample_frac_10 > subsample_frac_500
    assert subsample_frac_500 > subsample_frac_50k
    assert subsample_frac_50k > subsample_frac_1mil
    assert np.abs(subsample_frac_1mil - subsample_frac_100mil) < 10


# Test get_hyperparameters
def test_get_hyperparameters():
    k_range = (15, 150)
    la_res_range = (25, 175)
    k, la_res, metric = es.tl._clustering.get_hyperparameters(k_range, la_res_range)
    assert k_range[0] <= k <= k_range[1]
    assert la_res_range[0] <= la_res <= la_res_range[1]
    assert metric in ["euclidean", "cosine"]


def test_get_hyperparameters_random_seed():
    random.seed(42)
    k_range = (15, 150)
    la_res_range = (25, 175)
    k1, la_res1, metric1 = es.tl._clustering.get_hyperparameters(k_range, la_res_range)

    random.seed(42)
    k2, la_res2, metric2 = es.tl._clustering.get_hyperparameters(k_range, la_res_range)

    assert k1 == k2
    assert la_res1 == la_res2
    assert metric1 == metric2


# Test run_pca_dim_reduction
def test_run_pca_dim_reduction(X):
    X_pca = es.tl._prune_features.run_pca_dim_reduction(X)
    assert X_pca.shape[1] < X.shape[1]
    assert X_pca.shape[0] == X.shape[0]


# Test run_base_clustering
def test_run_la_clustering(X):
    k = 15
    la_res = 1.0
    result = es.tl._clustering.run_la_clustering(X, k, la_res)
    assert isinstance(result, np.ndarray)
    assert result.shape[1] == X.shape[0]


@pytest.fixture
def hyperparams_ls():
    return [None, (25, 175), None, False]


@pytest.fixture
def args_in(zarr_loc_static, hyperparams_ls):
    sparse = False
    random_seed = None
    process_id = 0
    return [zarr_loc_static, hyperparams_ls, sparse, random_seed, process_id]


def test_run_base_clustering_valid_input(args_in):
    result = es.tl._clustering.run_base_clustering(args_in)
    assert isinstance(result, coo_matrix)


# Test get_hard_soft_clusters
@pytest.fixture
def bipartite_graph_array():
    row = np.array([0, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9])
    col = np.array([0, 0, 1, 0, 1, 2, 2, 2, 3, 3, 3, 3])
    data = np.ones_like(row, dtype=int)
    bipartite = coo_matrix((data, (row, col)), shape=(np.max(row) + 1, np.max(col) + 1))
    return bipartite


@pytest.fixture
def setup_data(bipartite_graph_array):
    n = np.max(bipartite_graph_array.row) + 1
    clustering = np.array([0, 0, 1, 2])  # this is cluster assigns of the base clusters
    edges = np.concatenate(
        (
            np.expand_dims(bipartite_graph_array.row, axis=1),
            np.expand_dims(bipartite_graph_array.col + n, axis=1),
        ),
        axis=1,
    )
    bg = Graph(edges).as_undirected()
    type_ls = [0] * n
    type_ls.extend([1] * (bg.vcount() - n))
    bg.vs["type"] = type_ls
    return n, clustering, bg


def test_get_hard_soft_clusters(setup_data):
    n, clustering, bg = setup_data
    hard_clusters, soft_membership_matrix = es.tl._clustering.get_hard_soft_clusters(
        n, clustering, bg
    )

    soft_membership_matrix = soft_membership_matrix.toarray()

    # Test hard cluster assignments
    assert len(hard_clusters) == n
    assert np.all(np.unique(hard_clusters) == np.array([0, 1, 2]))

    # Test soft membership matrix shape
    unique_clusters = len(np.unique(clustering))
    assert soft_membership_matrix.shape == (n, unique_clusters)

    # Test soft membership matrix properties
    assert np.allclose(soft_membership_matrix.sum(axis=1), 1)
    assert np.all(soft_membership_matrix >= 0)


def test_get_hard_soft_clusters_single_cluster(setup_data):
    n, clustering, bg = setup_data
    clustering = np.zeros(clustering.shape[0])
    hard_clusters, soft_membership_matrix = es.tl._clustering.get_hard_soft_clusters(
        n, clustering, bg
    )

    soft_membership_matrix = soft_membership_matrix.toarray()
    
    # Test hard cluster assignments
    assert np.all(hard_clusters == 0)

    # Test soft membership matrix
    assert np.allclose(soft_membership_matrix, np.ones((n, 1)))


# Test consensus_cluster_leiden
def test_consensus_cluster_leiden(bipartite_graph_array):
    bipartite = bipartite_graph_array
    n = np.max(bipartite_graph_array.row) + 1
    in_args = (n, 1.0, bipartite)
    (
        hard_clusters,
        soft_membership_matrix,
        resolution,
    ) = es.tl._clustering.consensus_cluster_leiden(in_args)

    #assert isinstance(hard_clusters, pd.Categorical)
    assert len(hard_clusters) == n
    assert isinstance(soft_membership_matrix, csr_matrix)
    assert soft_membership_matrix.shape[0] == n
    assert soft_membership_matrix.shape[1] >= np.unique(hard_clusters).shape[0]
    assert np.allclose(soft_membership_matrix.sum(axis=1), 1.0)
    assert resolution == 1.0

# Test ensemble function
@pytest.fixture
def ensemble_args(zarr_loc_static):
    return {
        "zarr_loc": zarr_loc_static,
        "reduction": "pca",
        "metric": None,
        "ensemble_size": 3,  # Small size for testing
        "k_range": (15, 150),
        "la_res_range": (25, 175),
        "nprocs": 1,
        "sparse": False
    }

def test_ensemble(ensemble_args):
    result = es.tl.main.ensemble(**ensemble_args)
    assert isinstance(result, coo_matrix)
    
    # The shape should be (n_cells, n_clusters_total)
    z1 = zarr.open(ensemble_args["zarr_loc"], mode="r")
    n_cells = z1["X"].shape[0]
    assert result.shape[0] == n_cells
    
    # There should be at least one cluster for each member in the ensemble
    assert result.shape[1] >= 3
    
# Test consensus function
@pytest.fixture
def consensus_args(bipartite_graph_array):
    n = np.max(bipartite_graph_array.row) + 1
    return {
        "n": n,
        "bg": bipartite_graph_array,
        "nprocs": 1
    }

def test_consensus(consensus_args):
    hard_clusters, soft_membership_matrix, all_clusterings = es.tl.main.consensus(**consensus_args)
    
    # Check hard clusters
    assert len(hard_clusters) == consensus_args["n"]
    assert isinstance(hard_clusters, np.ndarray)
    
    # Check soft membership matrix
    assert soft_membership_matrix.shape[0] == consensus_args["n"]
    assert np.allclose(soft_membership_matrix.sum(axis=1), 1.0)
    
    # Check all_clusterings
    assert isinstance(all_clusterings, np.ndarray)
    assert all_clusterings.shape[0] == consensus_args["n"]
    # Should have multiple resolutions tested
    assert all_clusterings.shape[1] > 1
    
# Test main consensus_cluster function
def test_consensus_cluster_basic(adata, zarr_loc):
    
    # Run the full pipeline with minimal parameters
    result_adata = es.tl.consensus_cluster(
        adata, 
        zarr_loc=zarr_loc,
        ensemble_size=3,  # Small for testing
        nprocs=1
    )
    
    # Check that results are added to adata object
    assert "hard_clusters" in result_adata.obs
    assert "soft_membership_matrix" in result_adata.obsm
    assert "uncertainty_score" in result_adata.obs
    assert "bipartite" in result_adata.obsm
    
    # Check shapes
    assert len(result_adata.obs["hard_clusters"]) == adata.shape[0]
    assert result_adata.obsm["soft_membership_matrix"].shape[0] == adata.shape[0]
    
    # Check that multiresolution results are not included by default
    assert "multiresolution_clusters" not in result_adata.obsm

def test_consensus_cluster_with_multires(adata, zarr_loc):
    
    # Run with return_multires=True
    result_adata = es.tl.consensus_cluster(
        adata, 
        zarr_loc=zarr_loc,
        ensemble_size=3,  # Small for testing
        nprocs=1,
        return_multires=True
    )
    
    # Check that multiresolution results are included
    assert "multiresolution_clusters" in result_adata.obsm


# TEST PLOTTING FUNCTIONS
def test_smm_heatmap_default(adata_with_results):
    # Test with default parameters
    es.pl.smm_heatmap(adata_with_results)


def test_smm_heatmap_custom(adata_with_results):
    # Test with custom parameters
    es.pl.smm_heatmap(
        adata_with_results,
        smm_cmap="viridis",
        feat_cmap="magma",
        show=False,
        output_path="data/heatmap.png",
    )


@pytest.mark.skip(reason="Haven't figured out how to get this working yet.")
def test_smm_heatmap_invalid_output_path(adata_with_results):
    # Test exception with invalid output path
    with pytest.raises(Exception):
        es.pl.smm_heatmap(adata_with_results, output_path="invalid/path")


def test_min_max_scaler_default():
    data = np.array([1, 2, 3, 4, 5])

    # Test default min and max
    scaled = es.pl.plotting.min_max_scaler(data)
    assert np.isclose(scaled[0], 0)
    assert np.isclose(scaled[-1], 1)


def test_slanted_orders():
    data = np.array([[2, 0, 0], [0, 1, 2], [1, 2, 1]])

    # Test default parameters
    order_dict = es.pl.plotting.slanted_orders(data)
    assert order_dict["rows"].tolist() == [2, 1, 0]
    assert order_dict["cols"].tolist() == [1, 2, 0]

    # Test order rows only
    order_dict = es.pl.plotting.slanted_orders(data, order_cols=False)
    assert order_dict["rows"].tolist() == [1, 2, 0]
    assert order_dict["cols"].tolist() == [0, 1, 2]

    # Test order cols only
    order_dict = es.pl.plotting.slanted_orders(data, order_rows=False)
    assert order_dict["rows"].tolist() == [0, 1, 2]
    assert order_dict["cols"].tolist() == [0, 2, 1]

    # Test no ordering
    order_dict = es.pl.plotting.slanted_orders(data, order_rows=False, order_cols=False)
    assert order_dict["rows"].tolist() == [0, 1, 2]
    assert order_dict["cols"].tolist() == [0, 1, 2]


def test_run_umap_default(adata):
    # Test default return
    es.pl.plotting.run_umap(adata)
    assert "X_umap" in adata.obsm


def test_run_umap_return_layout(adata):
    # Test return layout
    embedding = es.pl.plotting.run_umap(adata, return_layout=True)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[1] == 2

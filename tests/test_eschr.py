import pytest

import eschr as es

import os
import shutil
import random
import zarr
import anndata
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

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
    es.tl.clustering.make_zarr(adata, "data/test_data_static.zarr")
    return "data/test_data_static.zarr"
    
# TEST TOOL FUNCTIONS

# make_zarr
def test_make_zarr_default(adata):
    es.tl.clustering.make_zarr(adata, None)
    assert os.path.exists("data_store.zarr")
    shutil.rmtree("data_store.zarr")

def test_make_zarr_custom_path(adata, zarr_loc):
    es.tl.clustering.make_zarr(adata, zarr_loc)
    assert os.path.exists(zarr_loc)
    shutil.rmtree(zarr_loc)

def test_make_zarr_content(adata, zarr_loc):
    es.tl.clustering.make_zarr(adata, zarr_loc)
    z = zarr.open(zarr_loc)
    X = z['X']
    adata_X_coo = coo_matrix(adata.X)
    assert np.array_equal(X['row'][:], adata_X_coo.row)
    assert np.array_equal(X['col'][:], adata_X_coo.col)
    assert np.allclose(X['data'][:], adata_X_coo.data) #allow for rounding differences
    shutil.rmtree(zarr_loc)

# get_subsamp_size
def test_get_subsamp_size():
    # Test extreme small n
    n = 10
    subsample_size_10 = []
    for i in range(1000):
        subsample_size_10.append(es.tl.clustering.get_subsamp_size(n))
    subsample_frac_10 = np.mean(subsample_size_10)/n

    # Test with small n
    n = 500
    subsample_size_500 = []
    for i in range(1000):
        subsample_size_500.append(es.tl.clustering.get_subsamp_size(n))
    subsample_frac_500 = np.mean(subsample_size_500)/n
    
    # Test with medium n
    n = 50000 
    subsample_size_50k = []
    for i in range(1000):
        subsample_size_50k.append(es.tl.clustering.get_subsamp_size(n))
    subsample_frac_50k = np.mean(subsample_size_50k)/n

    # Test with large n
    n = 1000000
    subsample_size_1mil = []
    for i in range(1000):
        subsample_size_1mil.append(es.tl.clustering.get_subsamp_size(n))
    subsample_frac_1mil = np.mean(subsample_size_1mil)/n

    # Test extreme large n
    n = 100000000
    subsample_size_100mil = []
    for i in range(1000):
        subsample_size_100mil.append(es.tl.clustering.get_subsamp_size(n))
    subsample_frac_100mil = np.mean(subsample_size_100mil)/n
    
    assert subsample_frac_10 > subsample_frac_500 
    assert subsample_frac_500 > subsample_frac_50k 
    assert subsample_frac_50k > subsample_frac_1mil 
    assert np.abs(subsample_frac_1mil - subsample_frac_100mil) < 10

# get_hyperparameters
def test_get_hyperparameters():
    k_range = (15, 150)
    la_res_range = (25, 175)
    k, la_res, metric = es.tl.clustering.get_hyperparameters(k_range, la_res_range)
    assert k_range[0] <= k <= k_range[1]
    assert la_res_range[0] <= la_res <= la_res_range[1]
    assert metric in ["euclidean", "cosine"]

def test_get_hyperparameters_random_seed():
    random.seed(42)
    k_range = (15, 150)
    la_res_range = (25, 175)
    k1, la_res1, metric1 = es.tl.clustering.get_hyperparameters(k_range, la_res_range)

    random.seed(42)
    k2, la_res2, metric2 = es.tl.clustering.get_hyperparameters(k_range, la_res_range)

    assert k1 == k2
    assert la_res1 == la_res2
    assert metric1 == metric2

# run_pca_dim_reduction
def test_run_pca_dim_reduction(X):
    X_pca = es.tl.clustering.run_pca_dim_reduction(X)
    assert X_pca.shape[1] < X.shape[1]
    assert X_pca.shape[0] == X.shape[0]

# run_base_clustering
def test_run_la_clustering(X):
    k = 15
    la_res = 1.0
    result = es.tl._leiden.run_la_clustering(X, k, la_res)
    assert isinstance(result, np.ndarray)
    assert result.shape[1] == X.shape[0]

@pytest.fixture
def hyperparams_ls():
    return [None, (25, 175), None]

@pytest.fixture
def args_in(zarr_loc_static, hyperparams_ls):
    return [zarr_loc_static, hyperparams_ls]

def test_run_base_clustering_valid_input(args_in):
    result = es.tl.clustering.run_base_clustering(args_in)
    assert isinstance(result, coo_matrix)

# get_hard_soft_clusters
@pytest.fixture
def setup_data():
    n = 10
    clustering = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    edges = [(0, 10), (1, 10), (1, 11), (2, 10), (2, 11), (3, 12), (4, 12), (5, 12), (6, 13), (7, 13), (8, 13), (9, 13)]
    bg = Graph.TupleList(edges, directed=False)
    return n, clustering, bg
    
# consensus_cluster_leiden


# TEST PLOTTING FUNCTIONS
def test_smm_heatmap_default(adata_with_results):
    # Test with default parameters
    es.pl.smm_heatmap(adata_with_results)

def test_smm_heatmap_custom(adata_with_results):
    # Test with custom parameters
    es.pl.smm_heatmap(adata_with_results, smm_cmap='viridis', feat_cmap='magma', 
                      show=False, output_path='data/heatmap.png')

@pytest.mark.skip(reason="Haven't figured out how to get this working yet.")
def test_smm_heatmap_invalid_output_path(adata_with_results):
    # Test exception with invalid output path
    with pytest.raises(Exception):
        es.pl.smm_heatmap(adata_with_results, output_path='invalid/path')

def test_min_max_scaler_default():
    data = np.array([1, 2, 3, 4, 5])
    
    # Test default min and max
    scaled = es.pl.plotting.min_max_scaler(data)
    assert np.isclose(scaled[0], 0)
    assert np.isclose(scaled[-1], 1)

def test_slanted_orders():
    data = np.array([[2,0,0], 
                 [0,1,2],
                [1,2,1]])
    
    # Test default parameters
    order_dict = es.pl.plotting.slanted_orders(data)
    assert order_dict['rows'].tolist() == [2,1,0]
    assert order_dict['cols'].tolist() == [1,2,0]

    # Test order rows only
    order_dict = es.pl.plotting.slanted_orders(data, order_cols=False)
    assert order_dict['rows'].tolist() == [1,2,0]
    assert order_dict['cols'].tolist() == [0,1,2]

    # Test order cols only
    order_dict = es.pl.plotting.slanted_orders(data, order_rows=False) 
    assert order_dict['rows'].tolist() == [0,1,2]
    assert order_dict['cols'].tolist() == [0,2,1]

    # Test no ordering
    order_dict = es.pl.plotting.slanted_orders(data, order_rows=False, order_cols=False)
    assert order_dict['rows'].tolist() == [0,1,2]
    assert order_dict['cols'].tolist() == [0,1,2]

def test_run_umap_default(adata):
    # Test default return 
    es.pl.plotting.run_umap(adata)
    assert 'X_umap' in adata.obsm

def test_run_umap_return_layout(adata):
    # Test return layout
    embedding = es.pl.plotting.run_umap(adata, return_layout=True)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[1] == 2

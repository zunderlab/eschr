import os
import traceback
import warnings
import zarr
from scipy.sparse import coo_matrix

def make_zarr_sparse(adata, zarr_loc):
    """
    Make zarr data store.

    Parameters
    ----------
    adata : `anndata.AnnData`
        AnnData object containing preprocessed data to be clustered in slot `.X`
    zarr_loc : str
        Path to save zarr store which will hold the data to be clustered.
    """
    print("making new zarr")
    if zarr_loc == None:
        zarr_loc = os.getcwd() + "/data_store.zarr"
    # Create or open the Zarr store
    z1 = zarr.open(zarr_loc, mode="w")

    data = coo_matrix(adata.X)
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


def make_zarr_dense(adata, zarr_loc):
    """
    Make zarr data store.

    Parameters
    ----------
    adata : `anndata.AnnData`
        AnnData object containing preprocessed data to be clustered in slot `.X`
    zarr_loc : str
        Path to save zarr store which will hold the data to be clustered.
    """
    print("making new zarr")
    if zarr_loc == None:
        zarr_loc = os.getcwd() + "/data_store.zarr"
    # Create or open the Zarr store
    z1 = zarr.open(zarr_loc, mode="w")

    row_chunks = min(5000, adata.X.shape[0])
    col_chunks = min(5000, adata.X.shape[1])
    chunks = (row_chunks, col_chunks)
    shape = (adata.X.shape[0], adata.X.shape[1])

    # Create the Zarr dataset
    zarr_dataset = z1.create_dataset("X", shape=shape, chunks=chunks, dtype="float32")

    # Write the data to Zarr in chunks
    for i in range(0, shape[0], chunks[0]):
        # Generate or load a chunk of data
        chunk_data = adata.X[
            i : i + chunks[0], :
        ]  # For example, replace with your actual data

        # Write the chunk to the appropriate slice
        zarr_dataset[i : i + chunks[0], :] = chunk_data

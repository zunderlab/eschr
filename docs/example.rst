Basic example script
====================

.. code-block:: python

    import eschr as es
    import pandas as pd
    import anndata

    # Read in data from a csv file.
    # The method expects features as columns
    # Use commented out ".T" if you have features as rows
    # Remove "index_col = 0" if your csv does not have row indices included.
    # Also ensure that data has already been preprocessed/scaled/normalized
    # as appropriate for your data type.
    data_filepath = "/path/to/your/data.csv"
    data = pd.read_csv(data_filepath, index_col=0)  # .T

    # Create AnnData object
    adata = anndata.AnnData(X=data)

    # Specify path for creating the zarr store that will be used for interacting with your data
    zarr_loc = "/path/to/data.zarr"

    # Run ESCHR consensus clustering
    # This function will return the AnnData object with ESCHR hard clsuter assignments,
    # soft cluster memberhsips, and uncertainty scores added.
    adata = es.tl.consensus_cluster(
        adata=adata,
        zarr_loc=zarr_loc,
        # nprocs=10 #optionally specify number of cores to use for multiprocesssing
    )

    # Plot soft membership matrix heatmap visualization
    es.pl.smm_heatmap(adata, output_path="/where/to/save/figure.png")

    # Plot umap visualization
    es.pl.umap(adata, output_path="/where/to/save/figure.png")

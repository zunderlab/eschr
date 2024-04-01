.. module:: eschr
.. automodule:: eschr
   :noindex:


API
===

Import eschr as::

   import eschr as es

Some convenient preprocessing functions translated from pagoda2 have been included:

Make zarr files
---------------

.. autosummary::
    :toctree: .

    readwrite.csv_to_zarr
    readwrite.make_zarr


Run consensus clustering
------------------------

.. autosummary::
    :toctree: .

    tl.consensus_cluster

Make plots
----------

.. autosummary::
    :toctree: .

    pl.smm_heatmap
    pl.umap

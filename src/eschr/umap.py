import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import umap

from .._prune_features import calc_highly_variable_genes, calc_pca
from . import _umap_utils

mpl.use("Agg")

sys.setrecursionlimit(1000000)

# flake8: noqa: RST210
# flake8: noqa: RST203
# flake8: noqa: E266
# flake8: noqa: B902


def run_umap(cc_obj, return_layout=False, n_neighbors=15, metric="euclidean", **kwargs):
    """
    Generate 2D UMAP embedding.

    UMAP (Uniform Manifold Approximation and Projection) is a manifold learning
    technique suitable for visualizing high-dimensional data. Besides tending to
    be faster than tSNE, it optimizes the embedding such that it best reflects
    the topology of the data, which we represent throughout Scanpy using a
    neighborhood graph. tSNE, by contrast, optimizes the distribution of
    nearest-neighbor distances in the embedding such that these best match the
    distribution of distances in the high-dimensional space.  We use the
    implementation of `umap-learn <https://github.com/lmcinnes/umap>`.
    Documentaion of UMAP parameters below is taken directly from umap
    package documentation.

    Parameters
    ----------
    cc_obj : :class:`sharc.ConsensusCluster`
        Filled object resulting from running SHaRC clustering.
    return_layout : bool, default False
        Whether to return layout. If false, layout will be added to cc_obj.
    n_neighbors : float (optional, default 15)
        The size of local neighborhood (in terms of number of neighboring
        sample points) used for manifold approximation. Larger values
        result in more global views of the manifold, while smaller
        values result in more local data being preserved. In general
        values should be in the range 2 to 100.
    metric: string or function (optional, default 'euclidean')
        The metric to use to compute distances in high dimensional space.
        If a string is passed it must match a valid predefined metric. If
        a general metric is required a function that takes two 1d arrays and
        returns a float can be provided. For performance purposes it is
        required that this be a numba jit'd function. Valid string metrics
        include:
        * euclidean
        * manhattan
        * chebyshev
        * minkowski
        * canberra
        * braycurtis
        * mahalanobis
        * wminkowski
        * seuclidean
        * cosine
        * correlation
        * haversine
        * hamming
        * jaccard
        * dice
        * russelrao
        * kulsinski
        * ll_dirichlet
        * hellinger
        * rogerstanimoto
        * sokalmichener
        * sokalsneath
        * yule
        Metrics that take arguments (such as minkowski, mahalanobis etc.)
        can have arguments passed via the metric_kwds dictionary. At this
        time care must be taken and dictionary elements must be ordered
        appropriately; this will hopefully be fixed in the future.
    **kwargs
        These parameters will be passed to the umap init function.

    Returns
    -------
    Depending on `return_layout`, returns or updates `cc_obj.adata`
    with the following fields.

    **X_umap** : `adata.obsm` field
        UMAP coordinates of data.
    """
    
    if cc_obj.adata.X.shape[1] > 6000:
        bool_features = calc_highly_variable_genes(cc_obj.adata.X)
        X = cc_obj.adata.X[:, bool_features]
    else:
        X = cc_obj.adata.X
    X_pca = np.array(calc_pca(X))
    ### FUNCTIONALITY FOR INITIAL POSITIONS WILL BE ADDED
    res = umap.UMAP(n_components=2, n_neighbors=n_neighbors, metric=metric, **kwargs).fit_transform(X_pca)
    if return_layout:
        return res
    else:
        cc_obj.adata.obsm["X_umap"] = res


def plot_umap(
    cc_obj, features=None, cat_palette="tab20", cont_palette="viridis", show=True, output_path=None, **kwargs
):
    """
    Make UMAP plot colored by hard clusters and confidence scores.

    Parameters
    ----------
    cc_obj : :class:`sharc.ConsensusCluster`
        Filled object resulting from running SHaRC clustering.
    features : list of str, default None
        Option to specify specific features to plot, if None then the method
        will calulate marker features for each cluster and plot those.
    cat_cmap : str, default 'tab20'
        Color map for categorical features.
    cont_cmap : str, default 'viridis'
        Color map for continuous features.
    show : bool, default True
        Whether to show the plot.
    output_path : str, default None
        Path specifying where to save the plot. If none, plot is not saved.
    **kwargs
        Args to pass along to matplotlib scatterplot.
    """
    # plt.rcParams['figure.figsize'] = [10, 8]
    # plt.rcParams['figure.dpi'] = 600 # 200 e.g. is really fine, but slower

    try:
        cc_obj.adata.obsm["X_umap"].shape[1]
    except Exception as e:
        print(e)
        if output_path is not None:
            try:
                print("No umap found - checking for existing umap layout file...")
                cc_obj.adata.obsm["X_umap"] = np.array(
                    pd.read_csv(os.path.join(("/").join(output_path.split("/")[0:-1]), "umap_layout.csv"))
                )
            except Exception as e:
                print(e)
                print("No umap found - running umap...")
                run_umap(cc_obj)
                pd.DataFrame(cc_obj.adata.obsm['X_umap']).to_csv(os.path.join(("/").join(output_path.split("/")[0:-1]), "umap_layout.csv"), index=None)
        else:
            print("No umap found - running umap...")
            run_umap(cc_obj)
    # For now plot_features is not available, needs troubleshooting
    plot_features = False
    if plot_features:
        if features is None:
            try:
                features = ["hard_clusters", "cell_conf_score"]
                features_to_plot = np.unique(cc_obj.adata.uns["rank_genes_groups"]["names"][0].tolist()).tolist()
                features.extend(features_to_plot)
            except Exception as e:
                print(e)
                print("Calculating hard cluster top marker genes for visualization")
                # log2 transform, if it is not already! (can check this my looking at max value in array)
                sc.tl.rank_genes_groups(cc_obj.adata, "hard_clusters", method="logreg")
                features = ["hard_clusters", "cell_conf_score"]
                features_to_plot = np.unique(cc_obj.adata.uns["rank_genes_groups"]["names"][0].tolist()).tolist()
                features.extend(features_to_plot)
        features_to_plot = ["hard_clusters", "cell_conf_score"]
        features_to_plot.extend(features)
    else:
        features_to_plot = ["hard_clusters", "cell_conf_score"]
    ("Done umap, generating figures...")
    plt.rcParams['figure.figsize'] = [10, 8]
    if output_path is not None:
        try:
            # sc.plt.umap(adata, color=features_to_plot, s=50, frameon=False, ncols=3, palette='tab20', save=output_path)
            # return_fig=True, show=False)
            fig = _umap_utils.embedding(
                cc_obj.adata,
                color=features_to_plot,
                frameon=False,
                ncols=2,
                palette=cat_palette,
                return_fig=True,
                show=False,
                **kwargs,
            )
            # with PdfPages(output_path) as pp:
            #    pp.savefig(fig)
            plt.savefig(output_path, bbox_inches="tight", pad_inches=0.05, dpi=600)
            if show:
                plt.show()
            else:
                for fig_x in fig:
                    plt.close(fig_x)
        except Exception as e:
            print(e)
    else:
        _umap_utils.embedding(cc_obj.adata, color=features_to_plot, frameon=False, ncols=2, palette=cat_palette, **kwargs)
        # palette=cluster_color_dict, edgecolor='none', size = 15, vmax=200)
        plt.show()

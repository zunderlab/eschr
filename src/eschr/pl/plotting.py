import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import umap
from scipy.cluster import hierarchy
from scipy.sparse import issparse
from scipy.spatial.distance import pdist

from ..tl._prune_features import calc_highly_variable_genes, calc_pca
from . import _umap_utils

mpl.use("Agg")  # this makes plt.show not work


sys.setrecursionlimit(1000000)

# flake8: noqa: RST210
# flake8: noqa: RST203
# flake8: noqa: F841
# flake8: noqa: B902
# flake8: noqa: E266


def smm_heatmap(
    adata,
    features=None,
    smm_cmap="gray_r",
    feat_cmap="YlOrBr",
    show=True,
    output_path=None,
):
    """
    Make a heatmap of soft cluster memberships.

    Parameters
    ----------
    adata : :class:`anndata.AnnData`
        AnnData object containing results from running ESCHR clustering.
    features : list of str, default None
        Option to specify specific features to plot, if None then the method
        will calulate marker features for each cluster and plot those.
    smm_cmap : str, default 'gray_r'
        Color map for the soft membership matrix heatmap.
    feat_cmap : str, default 'YlOrBr'
        Color map for the selected features heatmap.
    show : bool, default True
        Whether ot show the plot.
    output_path : str, default None
        Path specifying where to save the plot. If none, plot is not saved.
    """
    # Prep soft membership matrix data for plotting
    # Order rows by hclust or if too large by multidimensional sort
    if adata.obsm["soft_membership_matrix"].shape[0] <= 50000:
        row_order = hierarchy.dendrogram(
            hierarchy.linkage(
                pdist(adata.obsm["soft_membership_matrix"]), method="average"
            ),
            no_plot=True,
            color_threshold=-np.inf,
        )["leaves"]
    else:
        smm_with_index = np.insert(
            adata.obsm["soft_membership_matrix"],
            0,
            list(range(adata.obsm["soft_membership_matrix"].shape[0])),
            axis=1,
        )
        # Sort the list using sorted() function
        # and lambda function for multiple attributes
        sorted_list = sorted(
            smm_with_index.tolist(),
            key=lambda x: [x[i] for i in range(1, smm_with_index.shape[1])],
        )
        row_order = np.array(sorted_list)[:, 0].astype(int).tolist()
    # Re-order clusters to fall along the diagonal for easier visual interpretation
    row_col_order_dict = slanted_orders(
        adata.obsm["soft_membership_matrix"][row_order, :],
        order_rows=False,
        order_cols=True,
        squared_order=True,
        discount_outliers=True,
    )
    smm_reordered = adata.obsm["soft_membership_matrix"][row_order, :][
        row_col_order_dict["rows"].tolist(), :
    ]
    smm_reordered = smm_reordered[:, row_col_order_dict["cols"].tolist()]

    # For now plot_features is not enabled because it needs some troubleshooting
    plot_features = False
    if plot_features:
        plt.rcParams["figure.figsize"] = [15, 5]  # needs to adapt to number of features
        fig, (ax1, ax2) = plt.subplots(1, 2)
    else:
        plt.rcParams["figure.figsize"] = [10, 5]
        fig, ax1 = plt.subplots()

    heatmap = sns.heatmap(
        pd.DataFrame(smm_reordered, columns=row_col_order_dict["cols"].tolist()),
        cmap=smm_cmap,  # "Spectral_r" YlOrBr magma_r "viridis" #MAKE IT BE FLEXIBLE TO CATEGORICAL AND CONTINUOUS!!!!!
        cbar=True,
        cbar_kws=dict(use_gridspec=False, location="left"),
        xticklabels=True,
        yticklabels=False,
        ax=ax1,
    )

    if plot_features:
        # Prep annotation data for plotting
        if features == None:
            try:
                features = np.array(adata.uns["rank_genes_groups"]["names"][0].tolist())
            except Exception as e:
                print(e)
                print("Calculating hard cluster top marker genes for visualization")
                sc.tl.rank_genes_groups(adata, "hard_clusters", method="logreg")
                features = np.array(adata.uns["rank_genes_groups"]["names"][0].tolist())
                print("marker genes done")
        elif isinstance(features, list):
            features = np.array(features)
        elif isinstance(features, pd.core.series.Series):
            features = features.to_numpy()
        elif isinstance(features, np.ndarray):
            features = features
        else:
            raise Exception(
                "provided features must be in the form of a list, numpy array, or pandas series"
            )

        if issparse(adata.X):
            exprs_arr = adata.X[:, :].toarray()[row_order, :][
                row_col_order_dict["rows"].tolist(), :
            ]
        else:
            exprs_arr = adata.X[:, :][row_order, :][
                row_col_order_dict["rows"].tolist(), :
            ]
        print("exprs arr reordered")
        var_names = adata.var_names
        exprs_cols_ls = [
            exprs_arr[:, np.nonzero(var_names.astype(str) == x)[0][0]] for x in features
        ]
        print("exprs_cols_ls done")
        exprs_mat = pd.DataFrame(exprs_cols_ls).T
        exprs_mat = exprs_mat.reindex(
            columns=exprs_mat.columns[row_col_order_dict["cols"].tolist()]
        )
        exprs_mat.columns = features[row_col_order_dict["cols"].tolist()]
        print("reindex done")
        exprs_mat = exprs_mat.apply(min_max_scaler, axis=1)
        annotations_heatmap = sns.heatmap(
            pd.DataFrame(exprs_mat),
            cmap=feat_cmap,  # "Spectral_r" YlOrBr magma_r "viridis" #MAKE IT BE FLEXIBLE TO CATEGORICAL AND CONTINUOUS!!!!!
            cbar=True,
            cbar_kws=dict(use_gridspec=False, location="right"),
            xticklabels=True,
            yticklabels=False,
            ax=ax2,
        )

        annotations_heatmap.set_xticklabels(
            annotations_heatmap.get_xticklabels(),
            rotation=30,
            horizontalalignment="right",
        )

    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0.05, dpi=600)
        if show:
            plt.show()
        else:
            plt.close(fig)
    else:
        plt.show()


def min_max_scaler(data_1d_vec, min_val=0, max_val=1):
    """
    Scale 1D vector between a min and max value.

    Parameters
    ----------
    data_1d_vec : array-like
        The 1D vector to scale.
    min_val : int, default 0
        Lower bound on range for scaling.
    max_val : int, default 1
        Lower bound on range for scaling.

    Returns
    -------
    array-like
        Data scaled to range specified by min_val and max_val.
    """
    x, y = min(data_1d_vec), max(data_1d_vec)
    scaled_data_1d_vec = (data_1d_vec - x) / (y - x) * (max_val - min_val) + min_val
    return scaled_data_1d_vec


def slanted_orders(
    data,
    order_rows=True,
    order_cols=True,
    squared_order=True,
    # same_order=FALSE,
    discount_outliers=True,
    max_spin_count=10,
):
    """
    Compute rows and columns orders moving high values close to the diagonal.

    For a matrix expressing the cross-similarity between two
    (possibly different) sets of entities, this produces better results than
    clustering. This is because clustering does not care about the order of
    each of two sub-partitions. That is, clustering is as happy with
    `[(2, 1), (4, 3)]` as it is with the more sensible `[(1, 2), (3, 4)]`.
    As a result, visualizations of similarities using naive clustering
    can be misleading. Adapted from the package 'slanter' in R:
    tanaylab.github.io/slanter/

    Parameters
    ----------
    data : array-like
        A rectangular matrix containing non-negative values.
    order_rows : bool
        Whether to reorder the rows.
    order_cols : bool
        Whether to reorder the columns.
    squared_order : bool
        Whether to reorder to minimize the l2 norm
        (otherwise minimizes the l1 norm).
    discount_outliers : bool
        Whether to do a final order phase discounting outlier values
        far from the diagonal.
    max_spin_count : int
        How many times to retry improving the solution before giving up.

    Returns
    -------
    Dictionary with two keys, `rows` and `cols`, which contain their
    respective ordering.
    """
    rows_count = data.shape[0]
    cols_count = data.shape[1]

    row_indices = np.array(range(rows_count))
    col_indices = np.array(range(cols_count))

    best_rows_permutation = row_indices
    best_cols_permutation = col_indices

    if (order_rows or order_cols) and np.min(data) >= 0:
        # stopifnot(min(data) >= 0)
        if squared_order:
            data = data * data

        def reorder_phase(
            data,
            best_rows_permutation,
            best_cols_permutation,
            row_indices,
            col_indices,
            rows_count,
            cols_count,
        ):  # figure out cleaner way to have it inherit scope
            rows_permutation = best_rows_permutation
            cols_permutation = best_cols_permutation
            spinning_rows_count = 0
            spinning_cols_count = 0
            was_changed = True
            error_rows = None
            error_cols = None
            while was_changed:
                was_changed = False

                if order_cols:
                    sum_indexed_cols = np.sum(
                        (data.T * row_indices).T, axis=0
                    )  # colSums(sweep(data, 1, row_indices, `*`))
                    sum_squared_cols = np.sum(data, axis=0)  # colSums(data)
                    sum_squared_cols[np.where(sum_squared_cols <= 0)] = 1
                    ideal_col_index = sum_indexed_cols / sum_squared_cols

                    ideal_col_index = ideal_col_index * (cols_count / rows_count)
                    new_cols_permutation = np.argsort(ideal_col_index)  # -1*
                    error = new_cols_permutation - ideal_col_index
                    new_error_cols = sum(error * error)
                    new_changed = any(new_cols_permutation != col_indices)
                    if error_cols is None or new_error_cols < error_cols:
                        error_cols = new_error_cols
                        spinning_cols_count = 0
                        best_cols_permutation = cols_permutation[new_cols_permutation]
                    else:
                        spinning_cols_count = spinning_cols_count + 1

                    if new_changed and spinning_cols_count < max_spin_count:
                        was_changed = True
                        data = data[:, new_cols_permutation]
                        cols_permutation = cols_permutation[new_cols_permutation]

                if order_rows:
                    sum_indexed_rows = np.sum(
                        (data * col_indices), axis=1
                    )  # multiplies col indices accross each col (col_indices[0] * data[:,0])
                    sum_squared_rows = np.sum(data, axis=1)
                    sum_squared_rows[np.where(sum_squared_rows <= 0)] = 1
                    ideal_row_index = sum_indexed_rows / sum_squared_rows

                    ideal_row_index = ideal_row_index * (rows_count / cols_count)
                    new_rows_permutation = np.argsort(-1 * ideal_row_index)
                    error = new_rows_permutation - ideal_row_index
                    new_error_rows = sum(error * error)
                    new_changed = any(new_rows_permutation != row_indices)
                    if error_rows is None or new_error_rows < error_rows:
                        error_rows = new_error_rows
                        spinning_rows_count = 0
                        # print(type(new_rows_permutation), new_rows_permutation.shape)
                        # return rows_permutation, new_rows_permutation
                        best_rows_permutation = rows_permutation[new_rows_permutation]
                    else:
                        spinning_rows_count = spinning_rows_count + 1

                    if new_changed and spinning_rows_count < max_spin_count:
                        was_changed = True
                        data = data[new_rows_permutation, :]
                        rows_permutation = rows_permutation[new_rows_permutation]

            return best_rows_permutation, best_cols_permutation

        best_rows_permutation, best_cols_permutation = reorder_phase(
            data,
            best_rows_permutation,
            best_cols_permutation,
            row_indices,
            col_indices,
            rows_count,
            cols_count,
        )

    return {"rows": best_rows_permutation, "cols": best_cols_permutation}


def run_umap(adata, return_layout=False, n_neighbors=15, metric="euclidean", **kwargs):
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
    adata : :class:`anndata.AnnData`
        AnnData object containing results from running ESCHR clustering.
    return_layout : bool, default False
        Whether to return layout. If false, layout will be added to adata.
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
    Depending on `return_layout`, returns or updates `adata`
    with the following fields.

    **X_umap** : `adata.obsm` field
        UMAP coordinates of data.
    """

    if adata.X.shape[1] > 6000:
        bool_features = calc_highly_variable_genes(adata.X)
        X = adata.X[:, bool_features]
    else:
        X = adata.X
    X_pca = np.array(calc_pca(X))
    ### FUNCTIONALITY FOR INITIAL POSITIONS WILL BE ADDED
    res = umap.UMAP(
        n_components=2, n_neighbors=n_neighbors, metric=metric, **kwargs
    ).fit_transform(X_pca)
    if return_layout:
        return res
    else:
        adata.obsm["X_umap"] = res


def umap_heatmap(
    adata,
    features=None,
    cat_palette="tab20",
    cont_palette="viridis_r",
    show=True,
    output_path=None,
    **kwargs,
):
    """
    Make UMAP plot colored by hard clusters and confidence scores.

    Parameters
    ----------
    adata : :class:`anndata.AnnData`
        AnnData object containing results from running ESCHR clustering.
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
        adata.obsm["X_umap"].shape[1]
    except Exception as e:
        print(e)
        if output_path is not None:
            try:
                print("No umap found - checking for existing umap layout file...")
                adata.obsm["X_umap"] = np.array(
                    pd.read_csv(
                        os.path.join(
                            ("/").join(output_path.split("/")[0:-1]), "umap_layout.csv"
                        )
                    )
                )
            except Exception as e:
                print(e)
                print("No umap found - running umap...")
                run_umap(adata)
                pd.DataFrame(adata.obsm["X_umap"]).to_csv(
                    os.path.join(
                        ("/").join(output_path.split("/")[0:-1]), "umap_layout.csv"
                    ),
                    index=None,
                )
        else:
            print("No umap found - running umap...")
            run_umap(adata)
    # For now specifying plot_features is not available, needs troubleshooting
    features_to_plot = ["hard_clusters", "uncertainty_score"]
    ("Done umap, generating figures...")
    plt.rcParams["figure.figsize"] = [10, 8]
    if output_path is not None:
        try:
            # sc.plt.umap(adata, color=features_to_plot, s=50, frameon=False, ncols=3, palette='tab20', save=output_path)
            # return_fig=True, show=False)
            fig = _umap_utils.embedding(
                adata,
                color=features_to_plot,
                frameon=False,
                ncols=2,
                palette=cat_palette,
                cmap="viridis_r",
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
        _umap_utils.embedding(
            adata,
            color=features_to_plot,
            frameon=False,
            ncols=2,
            palette=cat_palette,
            cmap="viridis_r",
            **kwargs,
        )
        # palette=cluster_color_dict, edgecolor='none', size = 15, vmax=200)
        plt.show()

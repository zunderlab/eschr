import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.sparse import issparse
from scipy.spatial.distance import pdist

mpl.use("Agg")  # this makes plt.show not work


sys.setrecursionlimit(1000000)

# flake8: noqa: F841
# flake8: noqa: B902
# flake8: noqa: E266


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


def make_smm_heatmap(cc_obj, features=None, smm_cmap="gray_r", feat_cmap="YlOrBr", show=True, output_path=None):
    """
    Make a heatmap of soft cluster memberships.

    Parameters
    ----------
    cc_obj : :class:`sharc.ConsensusCluster`
        Filled object resulting from running SHaRC clustering.
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
    hard_clust = np.unique(cc_obj.adata.obs["hard_clusters"])
    cc_obj.adata.obsm["soft_membership_matrix"] = cc_obj.adata.obsm["soft_membership_matrix"][:, hard_clust]
    row_order = hierarchy.dendrogram(
        hierarchy.linkage(pdist(cc_obj.adata.obsm["soft_membership_matrix"]), method="average"),
        no_plot=True,
        color_threshold=-np.inf,
    )["leaves"]
    row_col_order_dict = slanted_orders(
        cc_obj.adata.obsm["soft_membership_matrix"][row_order, :],
        order_rows=False,
        order_cols=True,
        squared_order=True,
        discount_outliers=True,
    )
    smm_reordered = cc_obj.adata.obsm["soft_membership_matrix"][row_order, :][row_col_order_dict["rows"].tolist(), :]
    smm_reordered = smm_reordered[:, row_col_order_dict["cols"].tolist()]

    # For now plot_features is not enabled because it needs soem troubleshooting
    plot_features = False
    if plot_features:
        plt.rcParams["figure.figsize"] = [15, 5]
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
                features = np.array(cc_obj.adata.uns["rank_genes_groups"]["names"][0].tolist())
            except Exception as e:
                print(e)
                print("Calculating hard cluster top marker genes for visualization")
                sc.tl.rank_genes_groups(cc_obj.adata, "hard_clusters", method="logreg")
                features = np.array(cc_obj.adata.uns["rank_genes_groups"]["names"][0].tolist())
                print("marker genes done")
        elif isinstance(features, list):
            features = np.array(features)
        elif isinstance(features, pd.core.series.Series):
            features = features.to_numpy()
        elif isinstance(features, np.ndarray):
            features = features
        else:
            raise Exception("provided features must be in the form of a list, numpy array, or pandas series")

        if issparse(cc_obj.adata.X):
            exprs_arr = cc_obj.adata.X[:, :].toarray()[row_order, :][row_col_order_dict["rows"].tolist(), :]
        else:
            exprs_arr = cc_obj.adata.X[:, :][row_order, :][row_col_order_dict["rows"].tolist(), :]
        print("exprs arr reordered")
        var_names = cc_obj.adata.var_names
        exprs_cols_ls = [exprs_arr[:, np.nonzero(var_names.astype(str) == x)[0][0]] for x in features]
        print("exprs_cols_ls done")
        exprs_mat = pd.DataFrame(exprs_cols_ls).T
        exprs_mat = exprs_mat.reindex(columns=exprs_mat.columns[row_col_order_dict["cols"].tolist()])
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
            annotations_heatmap.get_xticklabels(), rotation=30, horizontalalignment="right"
        )

    if output_path is not None:
        try:
            plt.savefig(output_path, bbox_inches="tight", pad_inches=0.05, dpi=600)
            if show:
                plt.show()
            else:
                plt.close(annotations_heatmap)
        except Exception as e:
            print(e)
            print("You must provide an directory path to output_dir if save_plot is True")
    else:
        plt.show()


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
            data, best_rows_permutation, best_cols_permutation, row_indices, col_indices, rows_count, cols_count
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
            data, best_rows_permutation, best_cols_permutation, row_indices, col_indices, rows_count, cols_count
        )

    return {"rows": best_rows_permutation, "cols": best_cols_permutation}

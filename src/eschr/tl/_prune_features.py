"""Feature selection and dimensionality reduction functions."""
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import issparse

#warnings.filterwarnings('ignore', message='*Note that scikit-learn's randomized PCA might not be exactly reproducible*')
warnings.filterwarnings("ignore")

def materialize_as_ndarray(a):
    """Convert distributed arrays to ndarrays."""
    if type(a) in (list, tuple):
        return tuple(np.asarray(arr) for arr in a)
    return np.asarray(a)


def _get_mean_var(X):
    # - using sklearn.StandardScaler throws an error related to
    #   int to long trafo for very large matrices
    # - using X.multiply is slower
    if True:
        mean = X.mean(axis=0)
        if issparse(X):
            mean_sq = X.multiply(X).mean(axis=0)
            mean = mean.A1
            mean_sq = mean_sq.A1
        else:
            mean_sq = np.multiply(X, X).mean(axis=0)
        # enforece R convention (unbiased estimator) for variance
        var = (mean_sq - mean**2) * (X.shape[0] / (X.shape[0] - 1))
    else:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler(with_mean=False).partial_fit(X)
        mean = scaler.mean_
        # enforce R convention (unbiased estimator)
        var = scaler.var_ * (X.shape[0] / (X.shape[0] - 1))
    return mean, var


def calc_highly_variable_genes(
    X,  # Union[np.array, spmatrix]
    min_disp=None,
    max_disp=None,
    min_mean=None,
    max_mean=None,
    n_top_genes=None,
    n_bins=20,
    flavor="seurat",
) -> np.array:
    """
    Calculate highly variable genes.

    Internal function for annotating highly variable genes. Adapted from Seurat
    to python by Scanpy, adapted here from Scanpy. Expects logarithmized (or
    otherwise scaled) data. Depending on `flavor`, this reproduces the
    R-implementations of Seurat or Cell Ranger. The normalized dispersion is
    obtained by scaling with the mean and standard deviation of the dispersions
    for genes falling into a given bin for mean expression of genes. This means
    that for each bin of mean expression, highly variable genes are selected.

    Parameters
    ----------
    X : :class:`~numpy.array` or :class:`~scipy.sparse.spmatrix`
        Data matrix of shape `n_obs` × `n_vars`. Rows correspond
        to cells and columns to genes.
    min_mean : `float`, optional (default: 0.0125)
        If `n_top_genes` unequals `None`, this and all other cutoffs for the means and the
        normalized dispersions are ignored.
    max_mean : `float`, optional (default: 3)
        If `n_top_genes` unequals `None`, this and all other cutoffs for the means and the
        normalized dispersions are ignored.
    min_disp : `float`, optional (default: 0.5)
        If `n_top_genes` unequals `None`, this and all other cutoffs for the means and the
        normalized dispersions are ignored.
    max_disp : `float`, optional (default: `None`)
        If `n_top_genes` unequals `None`, this and all other cutoffs for the means and the
        normalized dispersions are ignored.
    n_top_genes : `int` or `None`, optional (default: `None`)
        Number of highly-variable genes to keep.
    n_bins : `int`, optional (default: 20)
        Number of bins for binning the mean gene expression. Normalization is
        done with respect to each bin. If just a single gene falls into a bin,
        the normalized dispersion is artificially set to 1. You'll be informed
        about this if you set `settings.verbosity = 4`.
    flavor : `{'seurat', 'cell_ranger'}`, optional (default: 'seurat')
        Choose the flavor for computing normalized dispersion. In their default
        workflows, Seurat passes the cutoffs whereas Cell Ranger passes
        `n_top_genes`.

    Returns
    -------
    gene_subset
        bool indicating if a gene is in the highly variable set or not
    """
    print("extracting highly variable genes")

    if n_top_genes is not None and not all([min_disp is None, max_disp is None, min_mean is None, max_mean is None]):
        # logg.info
        print("If you pass `n_top_genes`, all cutoffs are ignored.")

    if min_disp is None:
        min_disp = 0.5
    if min_mean is None:
        min_mean = 0.0125
    if max_mean is None:
        max_mean = 3
    if max_disp is None:
        max_disp = np.inf

    X = np.expm1(X) if flavor == "seurat" else X
    mean, var = materialize_as_ndarray(_get_mean_var(X))
    # now actually compute the dispersion
    mean[mean == 0] = 1e-12  # set entries equal to zero to small value
    dispersion = var / mean
    if flavor == "seurat":  # logarithmized mean as in Seurat
        dispersion[dispersion == 0] = np.nan
        dispersion = np.log(dispersion)
        mean = np.log1p(mean)
    # all of the following quantities are "per-gene" here
    df = pd.DataFrame()
    df["means"] = mean
    df["dispersions"] = dispersion
    if flavor == "seurat":
        df["mean_bin"] = pd.cut(df["means"], bins=n_bins)
        disp_grouped = df.groupby("mean_bin")["dispersions"]
        disp_mean_bin = disp_grouped.mean()
        disp_std_bin = disp_grouped.std(ddof=1)
        # retrieve those genes that have nan std, these are the ones where
        # only a single gene fell in the bin and implicitly set them to have
        # a normalized disperion of 1
        one_gene_per_bin = disp_std_bin.isnull()
        gen_indices = np.where(one_gene_per_bin[df["mean_bin"].values])[0].tolist()
        if len(gen_indices) > 0:
            # logg.debug(
            # raise Exception(
            print(
                f"Gene indices {gen_indices} fell into a single bin: their "
                "normalized dispersion was set to 1.\n    "
                "Decreasing `n_bins` will likely avoid this effect."
            )
        # Circumvent pandas 0.23 bug. Both sides of the assignment have dtype==float32,
        # but there’s still a dtype error without “.value”.
        disp_std_bin[one_gene_per_bin.values] = disp_mean_bin[one_gene_per_bin.values].values
        disp_mean_bin[one_gene_per_bin.values] = 0
        # actually do the normalization
        df["dispersions_norm"] = (
            df["dispersions"].values - disp_mean_bin[df["mean_bin"].values].values  # use values here as index differs
        ) / disp_std_bin[df["mean_bin"].values].values
    elif flavor == "cell_ranger":
        from statsmodels import robust

        df["mean_bin"] = pd.cut(df["means"], np.r_[-np.inf, np.percentile(df["means"], np.arange(10, 105, 5)), np.inf])
        disp_grouped = df.groupby("mean_bin")["dispersions"]
        disp_median_bin = disp_grouped.median()
        # the next line raises the warning: "Mean of empty slice"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            disp_mad_bin = disp_grouped.apply(robust.mad)
            df["dispersions_norm"] = (
                df["dispersions"].values - disp_median_bin[df["mean_bin"].values].values
            ) / disp_mad_bin[df["mean_bin"].values].values
    else:
        raise ValueError('`flavor` needs to be "seurat" or "cell_ranger"')
    dispersion_norm = df["dispersions_norm"].values.astype("float32")
    if n_top_genes is not None:
        dispersion_norm = dispersion_norm[~np.isnan(dispersion_norm)]
        dispersion_norm[::-1].sort()  # interestingly, np.argpartition is slightly slower
        disp_cut_off = dispersion_norm[n_top_genes - 1]
        gene_subset = np.nan_to_num(df["dispersions_norm"].values) >= disp_cut_off
        # logg.debug(
        # raise Exception(
        print(f"the {n_top_genes} top genes correspond to a " f"normalized dispersion cutoff of {disp_cut_off}")
    else:
        dispersion_norm[np.isnan(dispersion_norm)] = 0  # similar to Seurat
        gene_subset = np.logical_and.reduce(
            (
                mean > min_mean,
                mean < max_mean,
                dispersion_norm > min_disp,
                dispersion_norm < max_disp,
            )
        )

    return gene_subset


def calc_simple_filter(X) -> np.array:  #: Union[np.ndarray, spmatrix]
    """
    Calculate expressed genes.

    Filters genes simply and crudely (and fairly slowly) based on how many
    cells have scaled counts greater than one.

    Parameters
    ----------
    X : :class:`~numpy.array` or `~scipy.sparse.csr`
        Data matrix of shape `n_obs` × `n_vars`. Rows correspond
        to cells and columns to genes.

    Returns
    -------
    gene_subset : :class:`~numpy.array` of bool
        Bool indicating if a gene is in the highly variable set or not.
    """
    percent_zeros = np.array((X <= 1.0).sum(0))[0] / X.shape[0] * 100
    gene_subset = np.where(percent_zeros <= 99)[0]  # drop any features with >99% zeros and 1s
    return gene_subset


def calc_pca(
    X,  #: Union[np.ndarray, spmatrix],
    n_comps: int = 50,
    zero_center: Optional[bool] = None,
    svd_solver: str = "auto",
    random_state: int = 0,  # Optional[Union[int, RandomState]]
    return_info: bool = False,
    dtype: str = "float32",
    copy: bool = False,
) -> np.ndarray:  # Union[AnnData, np.ndarray, spmatrix]
    """
    Run principal component analysis.

    Computes PCA coordinates, loadings and variance decomposition. Uses the
    implementation of *scikit-learn*.

    Parameters
    ----------
    X
        The data matrix of shape ``n_obs`` × ``n_vars``.
        Rows correspond to cells and columns to genes.
    n_comps
        Number of principal components to compute.
    zero_center
        If `True`, compute standard PCA from covariance matrix.
        If ``False``, omit zero-centering variables
        (uses :class:`~sklearn.decomposition.TruncatedSVD`),
        which allows to handle sparse input efficiently.
        Passing ``None`` decides automatically based on sparseness of the data.
    svd_solver
        Which SVD solver to use: the ARPACK wrapper in SciPy
        (:func:`~scipy.sparse.linalg.svds`), the randomized algorithm due to
        Halko (2009), or `auto` chooses automatically depending on the size
        of the problem.
    random_state
        Change to use different initial states for the optimization.
    dtype
        Numpy data type string to which to convert the result.

    Returns
    -------
    X_pca : :class:`scipy.sparse.spmatrix` or :class:`numpy.ndarray`
        PCA representation of data.
    """
    #if svd_solver in {"auto", "randomized"}:
    # Add logging infrastructure post-submission
    # logg.info(
    #print(
    #    "Note that scikit-learn's randomized PCA might not be exactly "
    #    "reproducible across different computational platforms. For exact "
    #    "reproducibility, choose `svd_solver='arpack'.` This will likely "
    #    "become the Scanpy default in the future."
    #)

    if X.shape[1] < n_comps:
        n_comps = X.shape[1] - 1
        # logg.debug(
        # raise Exception(
        print(f"reducing number of computed PCs to {n_comps} " f"as dim of data is only {X.shape[1]}")

    if zero_center is None:
        zero_center = not issparse(X)
    if zero_center:
        from sklearn.decomposition import PCA

        if issparse(X):
            # logg.debug(
            raise Exception(
                "    as `zero_center=True`, " "sparse input is densified and may " "lead to huge memory consumption",
            )
            X = X.toarray()  # Copying the whole adata_comp.X here, could cause memory problems
        else:
            X = X
        pca_ = PCA(n_components=n_comps, svd_solver=svd_solver, random_state=random_state)
    else:
        from sklearn.decomposition import TruncatedSVD
        # Add logging infrastructure post-submission
        # logg.debug(
        # raise Exception(
        #print(
        #    "    without zero-centering: \n"
        #    + "    the explained variance does not correspond to the exact statistical defintion\n"
        #    + "    the first component, e.g., might be heavily influenced by different means\n"
        #    + "    the following components often resemble the exact PCA very closely"
        #)
        pca_ = TruncatedSVD(n_components=n_comps, random_state=random_state)
    X_pca = pca_.fit_transform(X)

    if X_pca.dtype.descr != np.dtype(dtype).descr:
        X_pca = X_pca.astype(dtype)

    return X_pca

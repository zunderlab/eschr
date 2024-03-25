import warnings
import nmslib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin

# flake8: noqa: B902

warnings.filterwarnings("ignore")

class NMSlibTransformer(TransformerMixin, BaseEstimator):
    """Wrapper for using nmslib as sklearn's KNeighborsTransformer"""

    def __init__(self, n_neighbors=5, metric="euclidean", method="sw-graph", n_jobs=1):
        self.n_neighbors = n_neighbors
        self.method = method
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(self, X):
        """
        Set up the ANN model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data.

        Returns
        -------
        self : object
            Returns the fitted instance.
        """
        # self.n_samples_fit_ = X.shape[0]

        # see more metric in the manual
        # https://github.com/nmslib/nmslib/tree/master/manual
        try:
            space = {
                "euclidean": "l2",
                "cosine": "cosinesimil",
                "l1": "l1",
                "l2": "l2",
                "jaccard": "jaccard_sparse",
            }[self.metric]
            if space == "jaccard_sparse":
                # print('jaccard_sparse')
                self.nmslib_ = nmslib.init(method=self.method, space=space, data_type=nmslib.DataType.OBJECT_AS_STRING)
                self.nmslib_.addDataPointBatch(X)
                # Set index parameters
                # These are the most important ones
                M = self.n_neighbors  # 30
                efC = self.n_neighbors  # 100
                index_time_params = {"M": M, "efConstruction": efC, "post": 0}
                self.nmslib_.createIndex(index_time_params)
            elif isinstance(X, csr_matrix):
                self.nmslib_ = nmslib.init(
                    method=self.method, space="l2_sparse", data_type=nmslib.DataType.SPARSE_VECTOR
                )
                self.nmslib_.addDataPointBatch(X)
                indexParams = {"efConstruction": (self.n_neighbors + 1)}
                self.nmslib_.createIndex(indexParams)
            else:
                self.nmslib_ = nmslib.init(method=self.method, space=space)
                self.nmslib_.addDataPointBatch(X)
                indexParams = {"efConstruction": (self.n_neighbors + 1)}
                self.nmslib_.createIndex(indexParams)
        except Exception as e:
            print("Error: " + e)
        return self

    def transform(self, X):
        """
        Find approximate nearest neighbors.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data.

        Returns
        -------
        indices : :class:`numpy.ndarray`
            Indices of nearest neighbors.
        distances : :class:`numpy.ndarray`
            Distances to nearest neighbors.
        """
        # n_samples_transform = X.shape[0]

        # For compatibility reasons, as each sample is considered as its own
        # neighbor, one extra neighbor will be computed.
        n_neighbors = self.n_neighbors + 1

        # Setting query-time parameters
        efS = n_neighbors  # 100
        query_time_params = {"efSearch": efS}
        self.nmslib_.setQueryTimeParams(query_time_params)
        results = self.nmslib_.knnQueryBatch(X, k=n_neighbors, num_threads=self.n_jobs)
        indices, distances = zip(*results)
        indices, distances = np.vstack(indices), np.vstack(distances)

        # indptr = np.arange(0, n_samples_transform * n_neighbors + 1,
        #                   n_neighbors)
        # kneighbors_graph = csr_matrix((distances.ravel(), indices.ravel(),
        #                               indptr), shape=(n_samples_transform,
        #                                               self.n_samples_fit_))

        return indices, distances  # kneighbors_graph

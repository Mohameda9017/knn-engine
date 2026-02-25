import numpy as np
from knn.backends._distances import _all_l2_dist2


def kneighbors_brute(X_train, x_query, k, metric="l2"):
    """
    Return the k nearest neighbors using an optimized brute-force search.

    This implementation supports both:
    - single query: x_query shape (d,)
    - batch queries: x_query shape (m, d)

    For each query point, it computes squared L2 distances to all training points (vectorized),
    uses partial selection (argpartition) to select k candidate neighbors without fully sorting
    all n distances, then sorts only the selected k candidates to return neighbors ordered from
    closest to farthest.

    Tie handling:
    - Within the returned set, neighbors are ordered deterministically by (distance, index).
    - Because argpartition performs partial selection, if multiple points share the same
        distance at the k-th boundary, the exact set of returned neighbors may differ from
        a full-sort oracle (though returned neighbors are still sorted consistently within
        the returned set).

    Parameters
    ----------
    X_train : np.ndarray of shape (n, d)
        Training data matrix where each row is a training point.
    x_query : np.ndarray of shape (d,) or (m, d)
        Query point(s). If 1D, treated as a single query. If 2D, treated as a batch of m queries.
    k : int
        Number of nearest neighbors to return. Must satisfy 1 <= k <= n.
    metric : str, default="l2"
        Distance metric to use. Supported in v1: "l2" (Euclidean).

    Returns
    -------
    indices : np.ndarray
        If x_query is (d,), returns shape (k,).
        If x_query is (m, d), returns shape (m, k).
        Each entry is an index into X_train. Rows are ordered by neighbor rank (closest -> farthest).
    distances : np.ndarray
        If x_query is (d,), returns shape (k,).
        If x_query is (m, d), returns shape (m, k).
        L2 distances aligned with `indices`.

    Raises
    ------
    ValueError
        If k is not in [1, n], if shapes are incompatible, or if `metric` is unsupported.
    TypeError
        If inputs are not numpy arrays or are not numeric.

    Time Complexity
    ---------------
    Let n = number of training points, d = number of features, m = number of queries.

    Per query:
    - Distance computation: O(n * d)
    - Partial selection (argpartition): ~O(n)
    - Sorting selected k candidates: O(k log k)

    Total for batch:
    - O(m * (n * d + n + k log k))
    Commonly summarized as:
    - O(m * (n * d + k log k)) when n*d dominates and k << n.

    Space Complexity
    ----------------
    - Outputs: O(m * k)
    - Per-query temporary storage:
        * dist2: O(n)
        * candidate list of size k: O(k)
    - (Not counting input storage for X_train / x_query)
    """
    # Validate inputs 
    if metric != "l2":
        raise ValueError("Metric must be l2")
    if not isinstance(X_train, np.ndarray):
        raise TypeError(f"X_train must be a numpy.ndarray; got {type(X_train).__name__}")
    if not isinstance(x_query, np.ndarray):
        raise TypeError(f"x_query must be a numpy.ndarray; got {type(x_query).__name__}")
    if not np.isfinite(X_train).all():
        raise ValueError("X_train contains NaN or infinite values.")
    if not np.isfinite(x_query).all():
        raise ValueError("x_query contains NaN or infinite values.")
    if X_train.ndim != 2:
        raise ValueError("X_train must be of shape 2D")
    if not np.issubdtype(X_train.dtype, np.number):
        raise TypeError("X_train must be numeric (int or float).")
    if not np.issubdtype(x_query.dtype, np.number):
        raise TypeError("x_query must be numeric (int or float).")


    n = X_train.shape[0]
    d = X_train.shape[1]

    # Converting all single queries to 2D. 
    single_query = (x_query.ndim == 1)
    if single_query: 
        X_query = x_query[None, :]
    elif x_query.ndim == 2:
        X_query = x_query
    else:
        raise ValueError("x_query must be 1D or 2D")

    if n == 0:
        raise ValueError("X_train must not be empty ")
    if d == 0:
        raise ValueError("Features must not be empty")
    if X_query.shape[1] != d:
        raise ValueError(f'x_query must be of shape({d},) or shape (m,{d})')
    if X_query.shape[0] == 0:
        raise ValueError("Batch must not be empty")
    if 0 >= k or k > n:
        raise ValueError("k needs to be 0<k<=n")

    m = X_query.shape[0]
    output_indices = np.empty((m, k), dtype=np.int64)
    output_distances = np.empty((m, k), dtype=np.float64)
    for i in range(m):
        # computing the distance between the training points and the query point and storing (distance, indx)
        dist2=_all_l2_dist2(X_train, X_query[i])
        partition = np.argpartition(dist2, k-1) # this will give us the indices of the k smallest distances but not necessarily sorted
        k_smallest_indices = partition[:k] # this will give us the indices of the k smallest distances but not necessarily sorted 

        # combining the k closest indicies with their corresponding distances into a list of tuples (distance, index) for sorting
        indices_distances = []
        for indx in k_smallest_indices:
            dist = dist2[indx]
            indices_distances.append((dist, indx))

        # sorting to get closest -> fartherest based on distance 
        indices_distances.sort() # Since we use argpartition, this becomes O(k log k), very fast when k is small 

        # storing the sorted indicies and distances of the k nearest neighbors into separate lists and taking sqrt of the distances to get actual l2 distance
        for j in range(k):
            dist, idx = indices_distances[j]
            dist_sqrt = np.sqrt(dist)
            output_distances[i, j] = dist_sqrt
            output_indices[i,j] = idx

    if single_query:
        return output_indices[0], output_distances[0]
    
    return output_indices, output_distances







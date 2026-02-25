import numpy as np

def kneighbors_brute_oracle(X_train, x_query, k, metric="l2"):
    """
    Reference (oracle) brute-force k-NN implementation used for testing/validation.

    Supports both:
    - single query: x_query shape (d,)
    - batch queries: x_query shape (m, d)

    For each query point, this oracle computes the (squared) L2 distance to every training point,
    then performs a FULL sort over all n distances to select the k nearest neighbors. This makes
    it slower than optimized backends (e.g., argpartition-based), but it is simple and
    deterministic—ideal as a correctness ground truth.

    Tie handling:
    - Neighbors are ordered deterministically by (distance, index).
        If two training points have the same distance, the smaller training index comes first.

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

    For each query:
    - Distance computation: O(n * d)
    - Full sort of n distances: O(n log n)

    Total for batch:
    - O(m * (n * d + n log n))

    Space Complexity
    ----------------
    - Output storage: O(m * k)
    - Per-query temporary list of (distance, index) pairs: O(n)
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
    out_indices = np.empty((m, k), dtype=np.int64)
    out_distances = np.empty((m, k), dtype=np.float64)
    
    # Looping through each query in the batch 
    for i in range(m):
        # computing the distance between the training points and the query point and storing (distance, indx)
        indices_distances = []
        for j,row in enumerate(X_train):
            distance = np.sum(((row - X_query[i]) **2 )) # will take sqrt of only the k nearest neighbors later
            indices_distances.append((distance, j)) 

        # sorting to get closest -> fartherest based on distance 
        indices_distances.sort()

        # extracting the indices and distances of the k nearest neighbors and taking sqrt of the distances to get actual l2 distance
        for j in range(k):
            dist, idx = indices_distances[j]
            dist_sqrt = np.sqrt(dist)
            out_indices[i, j] = idx
            out_distances[i,j] = dist_sqrt

    # allows single query to have shape of (k,)
    if single_query:
        return out_indices[0], out_distances[0]
    
    return out_indices, out_distances






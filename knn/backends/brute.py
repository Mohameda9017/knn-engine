import numpy as np

def kneighbors_brute(X_train, x_query, k, metric="l2"):
    """
    Return the k nearest neighbors of a SINGLE query point using brute-force search.

    This function computes the distance from `x_query` to every training point in `X_train`,
    then returns the indices of the k smallest distances (nearest neighbors). The returned
    neighbors are ordered from closest to farthest. If two training points have the same
    distance to `x_query`, ties are broken by choosing the smaller training index first.

    Parameters
    ----------
    X_train : np.ndarray of shape (n, d)
        Training data matrix where each row is a training point.
    x_query : np.ndarray of shape (d,)
        One query point to search neighbors for.
    k : int
        Number of nearest neighbors to return. Must satisfy 1 <= k <= n.
    metric : str, default="l2"
        Distance metric to use. Supported in v1: "l2" (Euclidean).
        (You can add "l1" later.)

    Returns
    -------
    indices : np.ndarray of shape (k,)
        Indices into X_train of the k nearest neighbors to x_query, sorted from
        closest to farthest.
    distances : np.ndarray of shape (k,)
        Distances corresponding to `indices`, aligned and sorted in the same order.

    Raises
    ------
    ValueError
        If k is not in [1, n], if shapes are incompatible, or if `metric` is unsupported.
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
    if n == 0:
        raise ValueError("X_train must not be empty ")
    if d == 0:
        raise ValueError("Features must not be empty")
    if x_query.ndim != 1 or x_query.shape[0] != d:
        raise ValueError(f'x_query must be of shape({d},)')
    if 0 >= k or k > n:
        raise ValueError("k needs to be 0<k<=n")
     
    # computing the distance between the training points and the query point and storing (distance, indx)
    dist2=_all_l2_dist2(X_train, x_query)
    indices_distances = [(d2,j) for j,d2 in enumerate(dist2)]



    # sorting to get closest -> fartherest based on distance 
    indices_distances.sort()

    indices = []
    distances = []
    for i in range(k):
        dist, idx = indices_distances[i]
        dist_sqrt = np.sqrt(dist)
        distances.append(dist_sqrt)
        indices.append(idx)

    return np.asarray(indices), np.asarray(distances)





# private helper function to calculate sqaured l2 distances but vectorized
def _all_l2_dist2(X_train, x_query):
    diff = X_train - x_query # this will broadcast x_query across all rows of X_train and give us a matrix of shape (n,d) where each row is the difference between the training point and the query point
    dist2 = np.sum(diff ** 2, axis=1) # this will sum the squared differences across the columns (features) for each row (training point) and give us a vector of shape (n,) where each element is the squared l2 distance from the query point to the corresponding training point
    return dist2

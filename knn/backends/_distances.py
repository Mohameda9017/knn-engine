import numpy as np 

# private helper function to calculate sqaured l2 distances but vectorized
def _all_l2_dist2(X_train, x_query):
    """
    Compute squared Euclidean (L2) distances from a single query point to all training points.

    This is a vectorized helper used by k-NN backends. It returns **squared** distances
    (no square root) because squared distances preserve neighbor ordering and are cheaper
    to compute. The caller can take `np.sqrt` only for the final selected neighbors if
    actual Euclidean distances are needed.

    Parameters
    ----------
    X_train : np.ndarray of shape (n, d)
        Training data matrix where each row is a point in d-dimensional space.
    x_query : np.ndarray of shape (d,)
        Single query point.

    Returns
    -------
    dist2 : np.ndarray of shape (n,)
        Squared L2 distances where `dist2[i] = ||X_train[i] - x_query||_2^2`.

    Notes
    -----
    - Uses NumPy broadcasting to compute all point-wise differences at once.
    - Runs in O(n * d) time for one query and n training points.
    - Does not perform input validation; callers are expected to validate shapes/types.
    """
    diff = X_train - x_query # this will broadcast x_query across all rows of X_train and give us a matrix of shape (n,d) where each row is the difference between the training point and the query point
    dist2 = np.sum(diff ** 2, axis=1) # this will sum the squared differences across the columns (features) for each row (training point) and give us a vector of shape (n,) where each element is the squared l2 distance from the query point to the corresponding training point
    return dist2

from knn.backends.brute_oracle import kneighbors_brute_oracle
from knn.backends.brute import kneighbors_brute
import numpy as np
import pytest


def test_oracle_returns_correct_indices_and_distances_on_simple_example():
    X_train = np.array([[0, 1], [7, 6], [7, 3], [5, 9]], dtype=float)
    x_query = np.array([3, 2], dtype=float)

    indices, distances = kneighbors_brute_oracle(X_train, x_query, k=2, metric="l2")

    assert indices.tolist() == [0, 2]
    assert np.allclose(distances, np.array([np.sqrt(10), np.sqrt(17)]))
    assert distances[0] <= distances[1]


def test_oracle_tie_breaks_by_lower_index_globally():
    # All three points [0,1] are tied at the smallest distance (sqrt(10)).
    # Oracle must pick the lowest indices among ties.
    X_train = np.array([[0, 1], [0, 1], [7, 3], [5, 9], [0, 1]], dtype=float)
    x_query = np.array([3, 2], dtype=float)

    indices, distances = kneighbors_brute_oracle(X_train, x_query, k=2, metric="l2")

    assert indices.tolist() == [0, 1]  # canonical: smallest indices among the tied group
    assert np.allclose(distances, np.array([np.sqrt(10), np.sqrt(10)]))
    assert indices.tolist() == sorted(indices.tolist())


def test_oracle_k_equals_1_returns_best_neighbor():
    X_train = np.array([[2, 2], [3, 3], [100, 100]], dtype=float)
    x_query = np.array([2, 2], dtype=float)

    indices, distances = kneighbors_brute_oracle(X_train, x_query, k=1, metric="l2")

    assert indices.tolist() == [0]
    assert np.allclose(distances, np.array([0.0]))


def test_oracle_k_equals_n_returns_all_neighbors_sorted():
    X_train = np.array([[0, 0], [2, 0], [1, 0]], dtype=float)
    x_query = np.array([0, 0], dtype=float)

    indices, distances = kneighbors_brute_oracle(X_train, x_query, k=3, metric="l2")

    assert indices.tolist() == [0, 2, 1]
    assert np.allclose(distances, np.array([0.0, 1.0, 2.0]))


def test_oracle_rejects_unsupported_metric():
    X_train = np.array([[1, 2], [3, 4]], dtype=float)
    x_query = np.array([1, 2], dtype=float)

    with pytest.raises(ValueError):
        kneighbors_brute_oracle(X_train, x_query, k=1, metric="l1")
    
def test_fast_matches_oracle_on_typical_random_data_no_ties():
    # With floats ties are unlikey
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((200,8))
    x_query = rng.standard_normal((8,))

    fast_indx, fast_dist = kneighbors_brute(X_train, x_query, 7, "l2")
    oracle_indx, oracle_dist = kneighbors_brute_oracle(X_train, x_query, 7, "l2")
    assert fast_indx.tolist() == oracle_indx.tolist()
    assert np.allclose(oracle_dist, fast_dist)


def test_fast_may_differ_from_oracle_on_kth_boundary_ties_but_remains_valid():
    # Construct a boundary tie:
    # - one point at distance 0
    # - four points at distance 1
    # Ask for k=3 -> oracle must pick lowest indices among the distance-1 ties.
    X_train = np.array(
        [
            [0, 0],   # idx 0: dist 0
            [1, 0],   # idx 1: dist 1
            [-1, 0],  # idx 2: dist 1
            [0, 1],   # idx 3: dist 1
            [0, -1],  # idx 4: dist 1
            [5, 5],   # idx 5: far away
        ],
        dtype=float,
    )
    x_query = np.array([0, 0], dtype=float)
    k = 3

    idx_fast, dist_fast = kneighbors_brute(X_train, x_query, k=k, metric="l2")
    idx_oracle, dist_oracle = kneighbors_brute_oracle(X_train, x_query, k=k, metric="l2")

    assert idx_oracle.tolist() == [0,1,2]
    assert np.allclose(dist_oracle, np.array([0.0, 1.0, 1.0]))

    # there must be the index 0 in fast brute
    assert 0 in set(idx_fast.tolist())

    # Fast must return k distinct indices
    assert len(idx_fast) == k
    assert len(set(idx_fast.tolist())) == k

    assert np.all(dist_fast[:-1] <= dist_fast[1:])

def test_oracle_returns_correct_indices_and_distances_on_batch_queries():
    X_train = np.array([[0, 1], [7, 6], [7, 3], [5, 9]], dtype=float)
    x_query = np.array([[3, 2], [1,2]], dtype=float)

    indicies, distances = kneighbors_brute_oracle(X_train, x_query, 2, 'l2')
    assert indicies.tolist() == [[0, 2], [0, 2]]
    assert np.allclose(distances, np.array([[np.sqrt(10), np.sqrt(17)], [np.sqrt(2), np.sqrt(37)]]))
    assert np.all(distances[:, 0] <= distances[:, 1])

def test_oracle_batch_equals_single_per_row():
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((60,5))
    x_query = rng.standard_normal((8,5))
    k = 6 
    indicies, distances = kneighbors_brute_oracle(X_train, x_query, k, 'l2')
    for i in range(x_query.shape[0]):
        indx, dist = kneighbors_brute_oracle(X_train, x_query[i], k, "l2")
        assert indicies[i].tolist() == indx.tolist()
        assert np.allclose(distances[i], dist)

def test_invariants_batch_outputs():
    rng = np.random.default_rng(2)
    X_train = rng.standard_normal((100, 6))
    X_query = rng.standard_normal((12, 6))
    for fn in (kneighbors_brute, kneighbors_brute_oracle):
        I, D = fn(X_train, X_query, 7, 'l2')
        m,k = I.shape[0], I.shape[1]
        # both the Index and distance vector must be of the same shape
        assert I.shape == (m,k)
        assert D.shape == (m,k)

        # in-bounds and distinct per row
        n = X_train.shape[0]
        assert np.all((I >= 0) & (I < n))
        for r in range(m):
            assert len(set(I[r].tolist())) == k

        # distances nondecreasing per row
        assert np.all(D[:, :-1] <= D[:, 1:] + 1e-12)
        # nonnegative
        assert np.all(D >= -1e-12)

def test_brute_equals_oracle_when_unique_kth_distance():
    rng = np.random.default_rng(3)
    X_train = rng.standard_normal((50,9))
    X_query = rng.standard_normal((9,9))
    k = 9
    I_brute, D_brute = kneighbors_brute(X_train, X_query, k, "l2")
    I_orc,  D_orc  = kneighbors_brute_oracle(X_train, X_query, k, "l2")

    for i in range(X_query.shape[0]):
        dist = np.sum((X_train - X_query[i])**2, axis = 1) # axis 1 allows us to sum across the columns of each row 
        s = np.sort(dist)
        if k < X_train.shape[0] and s[k-1] < s[k]: # meaning there is no distance ties
            assert I_brute[i].tolist() == I_orc[i].tolist()
            assert np.allclose(D_brute[i], D_orc[i])



def test_fast_valid_under_kth_boundary_ties():
    X_train = np.array(
    [
        [0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [5, 5]
    ],
    dtype=float,
    )
    X_query = np.array([[0, 0]], dtype=float)
    k = 3

    I_fast, D_fast = kneighbors_brute(X_train, X_query, k, "l2")
    I_orc,  D_orc  = kneighbors_brute_oracle(X_train, X_query, k, "l2")

    assert 0 in set(I_fast[0].tolist())
    # must not return something farther than oracle kth distance
    assert D_fast[0].max() <= D_orc[0, -1] + 1e-12
    # sorted by distance
    assert np.all(D_fast[0][:-1] <= D_fast[0][1:] + 1e-12) # all distances expect last one is less than all the distances except the first one. 


def test_rejects_k_out_of_range():
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((50,2))
    X_query = rng.standard_normal((2,))
    with pytest.raises(ValueError):
        kneighbors_brute(X_train, X_query, k=0, metric="l2")

from knn.backends.brute import kneighbors_brute
import numpy as np
import pytest

def test_handle_example_returns_distance_and_indices():
    X_train = np.array([[0,1], [7,6],[7,3],[5,9]])
    x_query = np.array([3,2])
    k = 2
    metric = 'l2'
    indices, distances = kneighbors_brute(X_train, x_query, k, metric)

    # Indices should be deterministic and exact
    assert indices.tolist() == [0,2]
    # Distances must be close
    assert np.allclose(distances, np.array([np.sqrt(10), np.sqrt(17)]))
    # Ordering of distances must be closest -> farthest
    assert distances[0] <= distances[1]

def test_tie_breaker_prefers_lower_index():
    # Two identical points at indices 0 and 1 have equal distance to the query
    X_train = np.array([[0,1], [0,1],[7,3],[5,9]])
    x_query = np.array([3,2])
    k = 2
    metric = 'l2'
    indices, distances = kneighbors_brute(X_train, x_query, k, metric)

    # Indices should prefer the lower index in case of a tie
    assert indices.tolist() == [0,1]
    assert np.allclose(distances, np.array([np.sqrt(10), np.sqrt(10)]))

def test_boundary_tie_returns_valid_neighbors_under_partial_selection():
    # Now 3 points have the same distance and k is going to be 2, so when using partiail selection, the returned indicies might change
    X_train = np.array([[0,1], [0,1],[7,3],[5,9], [0,1]])
    x_query = np.array([3,2])
    k = 2
    metric = 'l2'
    indices, distances = kneighbors_brute(X_train, x_query, k, metric)

    tied = [0,1,4]
    # Indices must come from the tied group (any 2 are valid under partial selection)
    assert set(indices.tolist()).issubset(tied)
    assert np.allclose(distances, np.array([np.sqrt(10), np.sqrt(10)]))

def test_k_equals_1_returns_single_best_neighbor():
    X_train = np.array([[2, 2], [3, 3], [100, 100]], dtype=float)
    x_query = np.array([2, 2], dtype=float)
    k = 1 
    metric = 'l2'
    indices, distances = kneighbors_brute(X_train, x_query, k, metric)

    assert len(indices.tolist()) == 1 
    assert indices.tolist() == [0]
    assert np.allclose(distances, np.array([0.0]))


def test_k_equals_n_returns_all_neighbors_sorted():
    X_train = np.array([[0, 0], [2, 0], [1, 0]], dtype=float)
    x_query = np.array([0, 0], dtype=float)

    indices, distances = kneighbors_brute(X_train, x_query, k=3, metric="l2")

    assert indices.tolist() == [0,2,1]
    assert np.allclose(distances, np.array([0.0, 1.0,2.0]))

def test_rejects_non_2d_X_train():
    X_train = np.array([0,1,2], dtype = float) # 1d X-train should be rejected 
    x_query = np.array([2,3,4], dtype=float)

    with pytest.raises(ValueError): # this test is supposed to raise a valueerror and if it does then pass
        kneighbors_brute(X_train, x_query, k=3, metric="l2")

def test_rejects_wrong_shape_x_query():
    X_train = np.array([[0, 0], [2, 0], [1, 0]], dtype=float)
    x_query = np.array([[2,3,2]], dtype=float) # x_query being shape 2D must be rejected 

    with pytest.raises(ValueError):
        kneighbors_brute(X_train, x_query, k=3, metric="l2")


def test_rejects_k_out_of_range():
    X_train = np.array([[0, 0], [2, 0], [1, 0]], dtype=float)
    x_query = np.array([2,3], dtype=float) 

    with pytest.raises(ValueError):
        kneighbors_brute(X_train, x_query, k=4, metric="l2")

    with pytest.raises(ValueError):
        kneighbors_brute(X_train, x_query, k=0, metric="l2")


def test_rejects_unsupported_distance_metric():
    X_train = np.array([[1, 2], [3, 4]], dtype=float)
    x_query = np.array([1, 2], dtype=float)

    with pytest.raises(ValueError):
        kneighbors_brute(X_train, x_query, k=1, metric="l1")
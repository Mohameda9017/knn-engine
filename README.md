# k-NN From Scratch (Engineering-Focused)

When going through the KNN algorithm, I knew that the naive-based approach did not scale well with larger datasets. 
So I wanted to explore how changing how we search for the nearest neighbors using different algorithms, such as brute force, KD-tree, and Ball tree, affects accuracy of its predicitions as well as performance. 

This project implements the **k-Nearest Neighbors (k-NN)** algorithm from scratch with a strong focus on
**software engineering, algorithmic clarity, and performance tradeoffs**.

The goal is not just to “use k-NN”, but to understand:
- how neighbor search actually works,
- why naive implementations are slow,
- and how different backends (brute force, trees, etc.) solve the same problem differently.

---

## Project Goals

- Implement k-NN step by step starting from first principles
- Separate **distance computation**, **neighbor search**, and **prediction**
- Compare different neighbor-search strategies:
  - Brute force
  - KD-tree
  - Ball tree
- Study performance tradeoffs as data size and dimensionality grow

This project is designed as a **software + machine learning** project.

---

## Current Status

✅ Project structure created  
✅ Brute-force neighbor search implemented for a **single query point**  
✅ Input validation and deterministic tie-breaking implemented  
✅ Vectorized distance computation (next)  
✅ k-selection optimization (avoid full sort)  
⏳ Tree-based backends (KD-tree, Ball tree)  
⏳ Benchmarks and comparisons  

---

## Core Function (v1)

### `kneighbors_brute`

Brute-force computation of nearest neighbors for **one query point**.

**Behavior**
- Computes the distance from the query point to all training points
- Selects the k smallest distances using partial selection
- Orders the returned neighbors from closest to farthest
- Note on ties: Because partial selection is used, when multiple points share the same distance at the k-th boundary, the exact set of returned neighbors may differ from what a full sort over all points would return. The ordering within the returned set is still deterministic.

**Inputs**
- `X_train`: array of shape `(n, d)`
- `x_query`: array of shape `(d,)`
- `k`: integer, `1 <= k <= n`
- `metric`: `"l2"` (Euclidean distance)

**Outputs**
- `indices`: array of shape `(k,)`
- `distances`: array of shape `(k,)`

This implementation serves as the **correctness baseline** for all future optimizations.

---

## Design Decisions

- **One query at a time (v1)**  
  Batch queries will be added later.

- **Euclidean distance only (v1)**  
  Keeps comparisons fair across backends.

- **Deterministic ordering (within the returned set)**
  After selecting k candidates, neighbors are sorted by distance and then by index to break ties consistently.

- **Performance-first implementation**  
  Uses vectorized distance computation and partial sorting to avoid an O(n log n) full sort when only k neighbors are needed.
---

## Planned Improvements

### Performance (Same Algorithm, Faster)
- Vectorized distance computation using NumPy
- Partial sorting / k-selection to avoid full `O(n log n)` sort

### Alternative Backends
- KD-tree (exact neighbor search, low-dimensional data)
- Ball tree (exact neighbor search, more flexible metrics)
- (Optional) Approximate nearest neighbors (ANN)

### Evaluation
- Runtime benchmarks vs data size and dimensionality
- Comparison with `scikit-learn`’s `KNeighborsClassifier`

---

## Why This Project

This project demonstrates:
- understanding of ML algorithms at a low level
- ability to design clean, extensible software
- awareness of algorithmic complexity and performance
- disciplined development: baseline → optimize → compare

It is intentionally built without relying on high-level ML libraries.

---

## How to Run (later)

Instructions will be added once batch prediction and benchmarks are implemented.

---

## Notes

This repository is built incrementally. Each optimization or backend is added only after
the previous version is correct and tested.

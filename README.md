# k-NN From Scratch (Engineering-Focused)

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
⏳ Vectorized distance computation (next)  
⏳ k-selection optimization (avoid full sort)  
⏳ Tree-based backends (KD-tree, Ball tree)  
⏳ Benchmarks and comparisons  

---

## Core Function (v1)

### `kneighbors_brute`

Brute-force computation of nearest neighbors for **one query point**.

**Behavior**
- Computes the distance from the query point to all training points
- Sorts by distance (ties broken by lower index)
- Returns the `k` closest neighbors

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

- **Deterministic tie-breaking**  
  If two points are equidistant, the smaller training index is preferred.

- **Correctness before optimization**  
  The brute-force method is implemented clearly before vectorization or k-selection.

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

import time
import numpy as np

from knn.backends.brute import kneighbors_brute


def _time_one(fn, repeats: int = 10, warmups: int = 3) -> float:
    """
    Returns median runtime (seconds) across `repeats`, after `warmups`.
    """
    # Warmup (helps reduce one-time overhead noise)
    for _ in range(warmups):
        fn()

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return float(np.median(times))


def bench_brute(
    ns=(1_000, 5_000, 10_000, 25_000, 50_000),
    d: int = 32,
    k: int = 10,
    repeats: int = 10,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)

    print("Benchmark: kneighbors_brute (one-query)")
    print(f"  d={d}, k={k}, repeats={repeats}")
    print("-" * 72)
    print(f"{'n':>10}  {'median_ms':>12}  {'ms_per_1k':>12}")
    print("-" * 72)

    for n in ns:
        if k > n:
            raise ValueError(f"k={k} cannot exceed n={n}")

        # Random numeric data
        X_train = rng.standard_normal((n, d), dtype=np.float64)
        x_query = rng.standard_normal((d,), dtype=np.float64)

        # Closure for timing
        def run():
            kneighbors_brute(X_train, x_query, k=k, metric="l2")

        sec = _time_one(run, repeats=repeats, warmups=3)
        ms = sec * 1000.0
        ms_per_1k = ms / (n / 1000.0)

        print(f"{n:>10}  {ms:>12.3f}  {ms_per_1k:>12.3f}")

    print("-" * 72)
    print("Tip: try changing d and n to see when distance compute vs sorting dominates.")


if __name__ == "__main__":
    # You can tweak these:
    bench_brute(
        ns=(1_000, 5_000, 10_000, 25_000, 50_000),
        d=32,
        k=10,
        repeats=10,
        seed=0,
    )

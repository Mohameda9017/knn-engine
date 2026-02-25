import time
import numpy as np

from knn.backends.brute_oracle import kneighbors_brute_oracle
# run with python -m benchmarks.bench_brute_oracle
# Benchmarking allows us to see how the runtime of your program grows as the input size increases. 
# Only measure what you care about, for us its the kneighbors_brute function. 
# Computers are noisy, so you run the benchmark multiple times and take the median, not the mean 

# Fixed values
d = 32
k = 10
metric = "l2"
ns = [1000, 5000, 10000, 25000, 50000]
repeats = 10
warmups = 3
m_batch = 64

# random number generator with a fixed seed 
# This is important because it allows us to keep reproducing the same inputs so differences refelect code changes and not input data 
rng = np.random.default_rng(0)

x_query = rng.standard_normal((d,))
X_query = rng.standard_normal((m_batch, d))

for n in ns:
    X_train = rng.standard_normal((n,d))

    ## Single query 

    # Warmup to prevent any startup noise 
    for i in range(warmups):
        kneighbors_brute_oracle(X_train, x_query, k, 'l2')

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        kneighbors_brute_oracle(X_train, x_query, k, metric)
        t1 = time.perf_counter()
        elapsed_s = (t1 - t0)
        times.append(elapsed_s)
    
    median_ms_single = np.median(times) * 1000.0

    ## Batch query 
    
    for i in range(warmups):
        kneighbors_brute_oracle(X_train, X_query, k, 'l2')

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        kneighbors_brute_oracle(X_train, X_query, k, metric)
        t1 = time.perf_counter()
        elapsed_s = (t1 - t0)
        times.append(elapsed_s)
    
    median_ms_batch = np.median(times) * 1000.0
    print(f"n={n:6d}  oracle_single_ms={median_ms_single:8.3f}  oracle_batch_ms={median_ms_batch:8.3f} (m={m_batch})")

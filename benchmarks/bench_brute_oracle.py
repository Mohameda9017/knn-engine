import time
import numpy as np

from knn.backends.brute_oracle import kneighbors_brute_oracle

# Benchmarking allows us to see how the runtime of your program grows as the input size increases. 
# Only measure what you care about, for us its the kneighbors_brute function. 
# Computers are noisy, so you run the benchmark multiple times and take the median, not the mean 

# Fixed values
d = 32
k = 10
metric = "l2"

ns = [1000, 5000, 10000, 25000, 50000] # Our input sizes 

# random number generator with a fixed seed 
# This is important because it allows us to keep reproducing the same inputs so differences refelect code changes and not input data 
rng = np.random.default_rng(0) 

for n in ns:
    X_train = rng.standard_normal((n,d))
    x_query = rng.standard_normal((d,))

    # Warmup to prevent any startup noise 
    kneighbors_brute_oracle(X_train, x_query, k=k, metric=metric)
    kneighbors_brute_oracle(X_train, x_query, k=k, metric=metric)
    kneighbors_brute_oracle(X_train, x_query, k=k, metric=metric)


    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        kneighbors_brute_oracle(X_train, x_query, k, metric)
        t1 = time.perf_counter()
        elapsed_s = (t1 - t0)
        times.append(elapsed_s)
    
    median_ms = np.median(times) * 1000.0
    print(f"n={n:6d}  median_ms={median_ms:8.3f}")

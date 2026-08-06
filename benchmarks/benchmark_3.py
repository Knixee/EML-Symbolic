import numpy as np
from sklearn.model_selection import train_test_split

from .draw_benchmarks import draw_benchmark_results
from .run_benchmark import run_benchmark

X = np.random.uniform(-1.5, 1.5, size=(2000, 2))
r = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
y = 1.0 / (r + 0.1)

params = {}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

print(f"\n==================================================")
print(f"RUNNING BENCHMARK: 1 / (sqrt(x1^2 + x2^2) + 0.1)")
print(f"==================================================")

y_pred = run_benchmark(X_train, y_train, X_test, y_test, params)

draw_benchmark_results(X_test, y_test, y_pred, title_name="1 / (sqrt(x1^2 + x2^2) + 0.1)")
